// Orbit camera: drag to rotate, wheel to zoom, optional auto-spin.

import { lookAt, multiply, normalize, perspective } from '../geometry/vec.js';

export class OrbitCamera {
  constructor(canvas) {
    this.yaw = 0;
    this.pitch = 0.2;
    this.radius = 3.5;
    this.spin = false;

    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      this.yaw -= (e.clientX - lastX) * 0.005;
      this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch + (e.clientY - lastY) * 0.005));
      lastX = e.clientX; lastY = e.clientY;
    });
    canvas.addEventListener('pointerup', (e) => {
      dragging = false;
      canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener('wheel', (e) => {
      this.radius = Math.max(1.3, Math.min(10, this.radius * Math.exp(e.deltaY * 0.001)));
    }, { passive: true });
  }

  eye() {
    const { yaw, pitch, radius: r } = this;
    return [r * Math.cos(pitch) * Math.sin(yaw), r * Math.sin(pitch), r * Math.cos(pitch) * Math.cos(yaw)];
  }

  /** Turn to look at the sphere point in direction d (keeps the distance). */
  lookToward(d) {
    const l = Math.hypot(...d);
    this.yaw = Math.atan2(d[0], d[2]);
    this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, Math.asin(d[1] / l)));
  }

  update() {
    if (this.spin) this.yaw -= 0.005;
  }

  viewProjection(aspect) {
    return multiply(perspective(Math.PI / 4, aspect, 0.1, 100), lookAt(this.eye(), [0, 0, 0], [0, 1, 0]));
  }

  /** Light from over the viewer's shoulder, so the visible hemisphere is lit. */
  lightDirection() {
    const e = this.eye();
    return normalize([e[0] + 0.5 * this.radius, e[1] + 0.8 * this.radius, e[2]]);
  }
}
