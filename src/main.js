// Entry point: set up WebGPU, build the initial state, and run the
// simulate-then-draw loop once per animation frame.

import { initDevice } from './gpu/device.js';
import { createIcosphere } from './geometry/icosphere.js';
import { generateInitialState } from './init/initial-state.js';
import { Simulation } from './sim/simulation.js';
import { simParamDefault } from './sim/params.js';
import { Renderer } from './render/renderer.js';
import { OrbitCamera } from './render/camera.js';

const MANTLE_SUBDIVISIONS = 3; // 642 mantle nodes

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

async function main() {
  const canvas = $('gpuCanvas');
  const { device, context, format } = await initDevice(canvas, reportError);
  const icosphere = createIcosphere(MANTLE_SUBDIVISIONS);
  const sim = new Simulation(device, icosphere, reportError);
  const renderer = new Renderer(device, context, format, sim, reportError);
  const camera = new OrbitCamera(canvas);
  const dtMyr = simParamDefault('dt_crust_myr');

  const urlSeed = Number.parseInt(new URLSearchParams(location.search).get('seed'), 10);
  if (Number.isFinite(urlSeed)) $('seed').value = urlSeed;

  let initStats = {};
  function regenerate() {
    const seed = Number.parseInt($('seed').value, 10) || 0;
    const t0 = performance.now();
    const state = generateInitialState(icosphere.vertices.length, { seed });
    sim.load(state);
    initStats = { ...state.stats, seed, ms: performance.now() - t0 };
  }
  $('regenerate').addEventListener('click', regenerate);
  regenerate();

  let lastStatsUpdate = 0;
  function frame(now) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

    camera.spin = $('spin').checked;
    camera.update();

    const encoder = device.createCommandEncoder();
    if (!$('pause').checked) sim.encodeStep(encoder);
    renderer.encodeDraw(encoder, {
      viewProj: camera.viewProjection(w / h),
      lightDir: camera.lightDirection(),
      showMantle: $('showMantle').checked,
    });
    device.queue.submit([encoder.finish()]);

    if (now - lastStatsUpdate > 250) {
      lastStatsUpdate = now;
      showStats([
        ['Seed', initStats.seed],
        ['Initial particles', initStats.particles],
        ['Initial bonds', initStats.bonds],
        ['Setup time', `${initStats.ms.toFixed(0)} ms`],
        ['Steps', sim.steps],
        ['Simulated time', `${(sim.steps * dtMyr).toFixed(2)} Myr`],
      ]);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((err) => reportError(err.message));
