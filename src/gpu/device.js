// WebGPU device setup and error reporting.

export async function initDevice(canvas, onError) {
  if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser (try a recent Chrome or Edge).');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found.');
  const device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (e) => onError(`GPU error: ${e.error.message}`));
  device.lost.then((info) => onError(`GPU device lost: ${info.message}`));

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });
  return { device, context, format };
}

/** Create a shader module and report compile errors with readable line numbers. */
export function createShaderModule(device, label, code, onError) {
  const module = device.createShaderModule({ label, code });
  module.getCompilationInfo().then((info) => {
    for (const m of info.messages) {
      const line = code.split('\n')[m.lineNum - 1] ?? '';
      const text = `[${label}] ${m.type} at line ${m.lineNum}: ${m.message}\n    ${line.trim()}`;
      if (m.type === 'error') onError(text); else console.warn(text);
    }
  });
  return module;
}
