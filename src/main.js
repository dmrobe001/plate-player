// Entry point: set up WebGPU, build the initial state, and run the
// simulate-then-draw loop once per animation frame, recording as configured.
//
// `window.plate` exposes the same operations for scripted use (tests, the
// browser console); see test/gpu.test.mjs.

import { initDevice } from './gpu/device.js';
import { SphereMesh } from './mantle/sphere-mesh.js';
import { Mantle, MANTLE_DEFAULTS } from './mantle/mantle.js';
import { generateInitialState, INIT_DEFAULTS } from './init/initial-state.js';
import { Simulation } from './sim/simulation.js';
import { Renderer } from './render/renderer.js';
import { OrbitCamera } from './render/camera.js';
import { Recorder, snapshotFromZip } from './record/recorder.js';
import { PlateTracker } from './plates/tracker.js';
import { BUILD_INFO } from './build-info.js';


const $ = (id) => document.getElementById(id);

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
  const canvas = $('gpuCanvas');
  const { device, context, format } = await initDevice(canvas, reportError);
  const mantleMesh = new SphereMesh(MANTLE_DEFAULTS.subdivisions);
  const sim = new Simulation(device, mantleMesh, reportError);
  const renderer = new Renderer(device, context, format, sim, reportError);
  const camera = new OrbitCamera(canvas);
  const plates = new PlateTracker(sim);
  const recorder = new Recorder(sim, {}, plates);

  const query = new URLSearchParams(location.search);
  const urlSeed = Number.parseInt(query.get('seed'), 10);
  if (Number.isFinite(urlSeed)) $('seed').value = urlSeed;
  if (query.has('paused')) $('pause').checked = true;

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

  function regenerate(seed = Number.parseInt($('seed').value, 10) || 0, options = {}) {
    return exclusive(async () => {
      $('seed').value = seed;
      const t0 = performance.now();
      const initOptions = { ...INIT_DEFAULTS, ...options, seed };
      const state = generateInitialState(initOptions);
      // The mantle runs alone first, so the crust never sees its artificial start.
      const mantle = new Mantle(seed, {}, mantleMesh);
      mantle.spinUp();
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
      sim.load(state, sim.mantle);
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
    regenerate, loadState, runSteps, restoreFromZip,
    setParams: (overrides) => exclusive(() => sim.setParams(overrides)),
    snapshot: () => exclusive(() => sim.snapshot()),
    restore: (snap) => exclusive(() => sim.restore(snap)),
    readBuffers: (names) => exclusive(() => sim.readBuffers(names)),
    exportRecord: () => exclusive(() => recorder.exportZip()),
    pause: (on = true) => { $('pause').checked = on; },
  };

  await regenerate();

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
      showNormal: $('showNormal').checked,
      showShear: $('showShear').checked,
      stressRefMPa: stressRef(),
    });
    device.queue.submit([encoder.finish()]);
    if (stepping) exclusive(() => captureIfDue()).catch((err) => reportError(err.message));

    if (now - lastStatsUpdate > 250) {
      lastStatsUpdate = now;
      const fmt = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : '–');
      showStats([
        ['Seed', initStats.seed],
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
        ['Mantle rms flow', `${sim.mantle.rmsSpeedCmYr().toFixed(1)} cm/yr`],
        ['Mantle plumes / drips born', `${sim.mantle.births.hot} / ${sim.mantle.births.cold}`],
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
