// WebGPU device setup and error reporting.

export const REQUIRED_STORAGE_BUFFERS = 9; // compute bindings in shaders/compute/bindings.wgsl

export async function initDevice(canvas, onError) {
  if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser (try a recent Chrome or Edge).');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found.');
  // The compute stage binds more storage buffers than WebGPU's default
  // limit of 8; desktop GPUs support many more, so ask for what the adapter has.
  const storageBuffers = adapter.limits.maxStorageBuffersPerShaderStage;
  if (storageBuffers < REQUIRED_STORAGE_BUFFERS) {
    throw new Error(`This GPU allows ${storageBuffers} storage buffers per shader stage; the simulation needs ${REQUIRED_STORAGE_BUFFERS}.`);
  }
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: storageBuffers } });
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
