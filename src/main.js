// Entry point: set up WebGPU, build the initial state, and run the
// simulate-then-draw loop once per animation frame, recording as configured.
//
// `window.plate` exposes the same operations for scripted use (tests, the
// browser console); see test/gpu.test.mjs.

import { initDevice } from './gpu/device.js';
import { SphereMesh } from './mantle/sphere-mesh.js';
import { Mantle, MANTLE_DEFAULTS } from './mantle/mantle.js';
import { generateInitialState, INIT_DEFAULTS } from './init/initial-state.js';
import { generateTwoPlateState, TWO_PLATE_DEFAULTS } from './init/two-plate.js';
import { KinematicMantle } from './mantle/kinematic.js';
import { Simulation } from './sim/simulation.js';
import { Renderer } from './render/renderer.js';
import { OrbitCamera } from './render/camera.js';
import { Recorder, snapshotFromZip } from './record/recorder.js';
import { PlateTracker } from './plates/tracker.js';
import { BUILD_INFO } from './build-info.js';
import { ColumnProfile, Edge, Particle, SLOTS, unpack } from './gpu/layout.js';
import * as profile from './columns/profile.js';
import * as law from './edges/law.js';
import * as flow from './columns/flow.js';
import * as area from './columns/area.js';
import * as volume from './columns/volume.js';
import * as shape from './columns/shape.js';
import * as topology from './edges/topology.js';
import { CAPACITY } from './sim/simulation.js';


const $ = (id) => document.getElementById(id);

/**
 * Read back particles, edges, the edge free list and slot lists, and check
 * them (src/edges/topology.js). Returns the unpacked arrays, the columns as
 * the edge kernels saw them (predicted positions), the edge set as pairKeys,
 * the overflow counters and a list of problems (empty = consistent).
 */
// Column profiles (design 4.0) read back, with the CPU reference's verdict.
async function inspectProfiles(sim) {
  const b = await sim.readBuffers(['particles', 'profiles']);
  const P = unpack(Particle, b.particles, CAPACITY.maxParticles);
  const G = unpack(ColumnProfile, b.profiles, CAPACITY.maxParticles);
  const active = [];
  for (let i = 0; i < CAPACITY.maxParticles; i++) if (P.is_active[i]) active.push(i);
  return { P, G, active, worst: profile.compareProfiles(P, G, sim.params, active) };
}

async function inspectEdges(sim) {
  const b = await sim.readBuffers(['particles', 'edges', 'edgePool', 'deltas', 'solver']);
  const P = unpack(Particle, b.particles, CAPACITY.maxParticles);
  const E = unpack(Edge, b.edges, CAPACITY.maxEdges);
  const cols = topology.columnsFrom(P, sim.params);
  const problems = topology.checkEdgeStorage({
    E, pool: new Uint32Array(b.edgePool), slots: new Uint32Array(b.deltas), layout: SLOTS, particleActive: P.is_active, cols,
  });
  const pairs = new Map();
  for (let e = 0; e < CAPACITY.maxEdges; e++) if (E.is_active[e]) pairs.set(topology.pairKey(E.particle_a[e], E.particle_b[e]), e);
  const solver = new Uint32Array(b.solver);
  return { P, E, cols, pairs, problems, edgePoolEmpty: solver[1], slotOverflow: solver[2] };
}

function reportError(message) {
  console.error(message);
  const box = $('errors');
  if (box.childElementCount < 10) box.append(Object.assign(document.createElement('div'), { textContent: message }));
}

function showStats(entries) {
  $('stats').replaceChildren(...entries.flatMap(([k, v]) => [
    Object.assign(document.createElement('dt'), { textContent: k }),
    Object.assign(document.createElement('dd'), { textContent: v }),
  ]));
}

function download(bytes, filename) {
  const url = URL.createObjectURL(new Blob([bytes], { type: 'application/zip' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

async function main() {
  const query = new URLSearchParams(location.search);
  // ?capacity=N: buffers for N particles (8 edge slots each) instead of the
  // planet-sized default. Small synthetic tests use it: under a software GPU
  // every dispatch and readback costs in proportion to capacity.
  const capacity = Number.parseInt(query.get('capacity'), 10);
  if (capacity > 0) Object.assign(CAPACITY, { maxParticles: capacity, maxEdges: 8 * capacity });
  // ?grid=N: hash grid cells per axis (default 32). Coarse test worlds with
  // few, large columns need larger cells (tessellation reach, design 4.8).
  const grid = Number.parseInt(query.get('grid'), 10);
  if (grid > 0) CAPACITY.hashGridDim = grid;
  const canvas = $('gpuCanvas');
  const { device, context, format } = await initDevice(canvas, reportError);
  const mantleMesh = new SphereMesh(MANTLE_DEFAULTS.subdivisions);
  const sim = new Simulation(device, mantleMesh, reportError);
  const renderer = new Renderer(device, context, format, sim, reportError);
  const camera = new OrbitCamera(canvas);
  const plates = new PlateTracker(sim);
  const recorder = new Recorder(sim, {}, plates);

  const urlSeed = Number.parseInt(query.get('seed'), 10);
  if (Number.isFinite(urlSeed)) $('seed').value = urlSeed;
  if (query.has('paused')) $('pause').checked = true;
  // ?world=random|two-plate: which world Regenerate builds (default: the two-plate test planet).
  if (['random', 'two-plate'].includes(query.get('world'))) $('world').value = query.get('world');

  // The simulation advances only when nothing asynchronous (a readback for
  // recording, a restore) is in flight, so every record is taken at an exact step.
  let busy = Promise.resolve();
  let isBusy = false;
  const exclusive = (fn) => {
    busy = busy.then(async () => {
      isBusy = true;
      try { return await fn(); } finally { isBusy = false; }
    });
    return busy;
  };

  let initStats = {};
  // The inputs update the recorder when edited; scripts may set
  // recorder.options directly.
  const syncRecorderOptions = () => {
    recorder.options.fieldIntervalMyr = Math.max(0, Number($('fieldInterval').value) || 0);
    recorder.options.snapshotIntervalMyr = Math.max(0, Number($('snapshotInterval').value) || 0);
  };
  $('fieldInterval').addEventListener('change', syncRecorderOptions);
  $('snapshotInterval').addEventListener('change', syncRecorderOptions);
  syncRecorderOptions();

  // One readback serves plate labelling and recording when either is due.
  async function captureIfDue(force = {}) {
    const recording = $('recordOn').checked;
    const due = recording ? recorder.due() : { fields: false, snapshot: false };
    const want = { fields: due.fields || !!force.fields, snapshot: due.snapshot || !!force.snapshot };
    const platesDue = plates.due() || want.fields;
    if (!platesDue && !want.fields && !want.snapshot) return;
    const snap = await sim.snapshot();
    if (platesDue) plates.update(snap);
    if (want.fields || want.snapshot) await recorder.capture(want, snap);
  }

  function regenerate(seed = Number.parseInt($('seed').value, 10) || 0, options = {}, world = $('world').value) {
    return exclusive(async () => {
      $('seed').value = seed;
      $('world').value = world;
      const t0 = performance.now();
      let state, mantle, initOptions;
      if (world === 'two-plate') {
        // Deterministic (no seed); its own parameters scale lengths with its column size.
        initOptions = { ...TWO_PLATE_DEFAULTS, ...options, world };
        state = generateTwoPlateState(initOptions);
        sim.resetParams(state.params);
        mantle = new KinematicMantle(state.mantle, mantleMesh);
        // Face the trench, turned toward plate A: the continent, the trench and the incoming microcontinent.
        const { q, normal: m } = state.frame;
        camera.lookToward(q.map((v, c) => -0.8 * v + 0.6 * m[c]));
      } else {
        initOptions = { ...INIT_DEFAULTS, ...options, seed, world };
        state = generateInitialState(initOptions);
        sim.resetParams({});
        // The mantle runs alone first, so the crust never sees its artificial start.
        mantle = new Mantle(seed, {}, mantleMesh);
        mantle.spinUp();
      }
      sim.load(state, mantle);
      initStats = { ...state.stats, seed, ms: performance.now() - t0 };
      recorder.reset({ seed, init_options: initOptions, init_stats: state.stats, restarted_from: null });
      plates.reset();
      await captureIfDue({ fields: true, snapshot: true });
    });
  }

  function restoreFromZip(bytes, sourceName = 'record') {
    return exclusive(async () => {
      const { snapshot, manifest } = snapshotFromZip(bytes);
      sim.restore(snapshot);
      const seed = manifest.run.seed;
      if (seed !== undefined) $('seed').value = seed;
      initStats = { ...(manifest.run.init_stats ?? {}), seed, ms: 0 };
      recorder.reset({
        ...manifest.run,
        restarted_from: { source: sourceName, build: manifest.model.build, step: snapshot.step, time_myr: snapshot.timeMyr },
      });
      plates.reset(); // plate ids restart from 1 after a resume
      await captureIfDue({ fields: true, snapshot: true });
    });
  }

  /** Load a hand-built state (same shape as generateInitialState's result). */
  function loadState(state, run = { seed: null, init_options: 'custom' }) {
    return exclusive(async () => {
      sim.load(state, sim.mantle ?? new Mantle(1, {}, mantleMesh)); // (?empty: an unspun mantle; tests turn the flow off)
      initStats = { ...(state.stats ?? {}), seed: run.seed, ms: 0 };
      recorder.reset({ ...run, restarted_from: null });
      plates.reset();
    });
  }

  /** Advance n steps without drawing (used by tests and scripts). */
  function runSteps(n) {
    return exclusive(async () => {
      // Keep at most two steps in flight: queuing many slow steps at once can
      // trip the browser's GPU watchdog, which kills the device.
      const inFlight = [];
      for (let i = 0; i < n; i++) {
        await sim.step();
        inFlight.push(device.queue.onSubmittedWorkDone());
        if (inFlight.length > 2) await inFlight.shift();
        await captureIfDue();
      }
      await Promise.all(inFlight);
    });
  }

  $('regenerate').addEventListener('click', () => regenerate());
  $('world').addEventListener('change', () => regenerate());
  $('snapshotNow').addEventListener('click', () => exclusive(() => recorder.capture({ fields: false, snapshot: true })));
  $('downloadRecord').addEventListener('click', () => exclusive(() => {
    download(recorder.exportZip(), `plate-player_seed${$('seed').value}_step${sim.steps}.zip`);
  }));
  $('loadRecord').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      await restoreFromZip(new Uint8Array(await file.arrayBuffer()), file.name);
    } catch (err) {
      reportError(`Could not resume from ${file.name}: ${err.message}`);
    }
    e.target.value = '';
  });

  // Log slider: 10^value MPa (1 .. 1000 MPa, default ~150 MPa = yield stress).
  const stressRef = () => 10 ** Number($('stressRef').value);
  const showStressRef = () => { $('stressRefOut').textContent = `${stressRef().toFixed(0)} MPa`; };
  $('stressRef').addEventListener('input', showStressRef);
  showStressRef();

  window.plate = {
    sim, recorder, plates, device, BUILD_INFO,
    layout: { Particle, Edge, ColumnProfile, SLOTS },          // struct offsets for reading raw buffers
    regenerate, loadState, runSteps, restoreFromZip,
    setParams: (overrides) => exclusive(() => sim.setParams(overrides)),
    snapshot: () => exclusive(() => sim.snapshot()),
    restore: (snap) => exclusive(() => sim.restore(snap)),
    readBuffers: (names) => exclusive(() => sim.readBuffers(names)),
    exportRecord: () => exclusive(() => recorder.exportZip()),
    pause: (on = true) => { $('pause').checked = on; },
    // Inspection (design 9, development strategy): CPU reference functions,
    // and a readback of the edge storage checked against its invariants.
    debug: {
      topology,
      edges: () => exclusive(() => inspectEdges(sim)),
      profile,
      profiles: () => exclusive(() => inspectProfiles(sim)),
      law,
      flow,
      area,
      volume,
      shape,
      counters: () => exclusive(() => sim.counters()),
    },
  };

  // ?empty: start with no world (synthetic tests load their own with loadState).
  if (!query.has('empty')) await regenerate();
  window.plate.ready = true;

  let lastStatsUpdate = 0;
  function frame(now) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

    camera.spin = $('spin').checked;
    camera.update();

    const stepping = !$('pause').checked && !isBusy;
    if (stepping) exclusive(() => sim.step()).catch((err) => reportError(err.message));
    const encoder = device.createCommandEncoder();
    renderer.encodeDraw(encoder, {
      viewProj: camera.viewProjection(w / h),
      lightDir: camera.lightDirection(),
      showMantle: $('showMantle').checked,
      colorByPlate: $('colorByPlate').checked,
      showForce: $('showForce').checked,
      showVelocity: $('showVelocity').checked,
      showStress: $('showStress').checked,
      stressRefMPa: stressRef(),
    });
    device.queue.submit([encoder.finish()]);
    if (stepping) exclusive(() => captureIfDue()).catch((err) => reportError(err.message));

    if (now - lastStatsUpdate > 250) {
      lastStatsUpdate = now;
      const fmt = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : '–');
      showStats([
        ['World', initStats.world ?? 'random'],
        ['Seed', initStats.world === 'two-plate' ? '– (deterministic)' : initStats.seed],
        ['Initial particles', initStats.particles ?? '–'],
        ['Bonds per particle', fmt(initStats.bondsPerParticle)],
        ['Continental fraction', fmt(initStats.continentalFraction)],
        ['Damaged bonds (>0.5)', fmt(initStats.damagedBondFraction)],
        ['Worst overlap after relaxing', typeof initStats.relaxResidual === 'number' ? `${(100 * initStats.relaxResidual).toFixed(1)}% of contact` : '–'],
        ['Apothem fit', typeof initStats.maxBondStrain === 'number' ? `max strain ${initStats.maxBondStrain.toExponential(0)}, size ${initStats.apothemRange.map((r) => r.toFixed(2)).join('–')}× nominal` : '–'],
        ['Coverage', typeof initStats.coverage === 'number' ? `${(100 * initStats.coverage).toFixed(0)}% of the sphere` : '–'],
        ['Setup time', `${fmt(initStats.ms, 0)} ms`],
        ['Steps', sim.steps],
        ['Simulated time', `${sim.timeMyr.toFixed(2)} Myr`],
        ['Mantle rms flow', sim.mantle ? `${sim.mantle.rmsSpeedCmYr().toFixed(1)} cm/yr` : '–'],
        ['Mantle plumes / drips born', sim.mantle ? `${sim.mantle.births.hot} / ${sim.mantle.births.cold}` : '–'],
        ['Plates', `${plates.summary.length} (${plates.summary.filter((p) => p.size >= 100).length} with ≥100 particles)`],
        ['Plate speed (median rms)', Number.isFinite(plates.median('rms_speed_mm_per_yr')) ? `${plates.median('rms_speed_mm_per_yr').toFixed(2)} mm/yr` : '–'],
        ['Non-rigid part', Number.isFinite(plates.median('rms_nonrigid_speed_mm_per_yr')) ? `${plates.median('rms_nonrigid_speed_mm_per_yr').toFixed(2)} mm/yr` : '–'],
        ['Plate events', plates.events.length],
        ['Recorded', `${recorder.samples.length} fields, ${recorder.snapshots.length} snapshots`],
        ['Record size', `${(recorder.sizeBytes / 2 ** 20).toFixed(1)} MiB`],
        ['Build', `${String(BUILD_INFO.commit).slice(0, 7)}${BUILD_INFO.dirty ? '+' : ''}`],
      ]);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((err) => reportError(err.message));
