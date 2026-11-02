// A prescribed (kinematic) mantle for controlled scenarios: the flow beneath
// two plates whose relative motion is a rotation about an Euler pole P lying
// on their shared boundary (src/init/two-plate.js).
//
// Frame: m, the unit normal of the boundary great circle (plate A on the
// x . m > 0 side, plate B on the other), and q = m x P. The boundary is the
// great circle through P and q; relative rotation of A about P carries A
// away from B where x . q > 0 (a ridge) and into B where x . q < 0 (a trench).
//
// Flow: u(x) = omega (P x x) S(s), s = asin(x . m) the signed angular
// distance from the boundary, S(s) = (1 + tanh(s / w)) / 2. Plate B's mantle
// is at rest (B is the reference frame), A's turns rigidly at omega, with a
// transition of width w at the boundary. P x x is divergence-free, so
// div u = omega (P x x) . grad S = omega S'(s) (x . q) / cos(s): upwelling
// along the ridge half, downwelling along the trench half, nowhere else.
// Nothing evolves: the same interface as Mantle (src/mantle/mantle.js), with
// a constant flow.

const CM_YR_TO_RAD_MYR = 10 / 6371;

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Flow and divergence at unit vector x (rad/Myr, 1/Myr). */
export function kinematicFlow(x, { pole, normal, omegaRadMyr, widthRad }) {
  const s = Math.asin(Math.max(-1, Math.min(1, dot(x, normal))));
  const S = 0.5 * (1 + Math.tanh(s / widthRad));
  const dS = 0.5 / widthRad / Math.cosh(s / widthRad) ** 2;
  const v = cross(pole, x);
  const q = cross(normal, pole);
  return {
    velocity: v.map((c) => omegaRadMyr * S * c),
    divergence: (omegaRadMyr * dS * dot(x, q)) / Math.max(Math.cos(s), 1e-6),
  };
}

export class KinematicMantle {
  /** options: {pole, normal (unit, orthogonal), omegaRadMyr, widthRad}. */
  constructor(options, mesh) {
    this.kind = 'kinematic';
    this.description = 'prescribed two-plate rotation about an Euler pole on the plate boundary (src/mantle/kinematic.js)';
    this.options = { ...options };
    this.mesh = mesh;
    const n = mesh.n;
    this.velocity = new Float64Array(3 * n);
    this.vComp = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
    this.divergence = new Float64Array(n);
    this.b = new Float64Array(n); // drawn: divergence normalised to [-1, 1] (orange = upwelling)
    this.scale = 1;
    this.time = 0;
    this.births = { hot: 0, cold: 0 };
    let maxDiv = 0;
    for (let i = 0; i < n; i++) {
      const { velocity, divergence } = kinematicFlow(mesh.point(i), this.options);
      for (let d = 0; d < 3; d++) this.velocity[3 * i + d] = this.vComp[d][i] = velocity[d];
      this.divergence[i] = divergence;
      maxDiv = Math.max(maxDiv, Math.abs(divergence));
    }
    for (let i = 0; i < n; i++) this.b[i] = maxDiv > 0 ? this.divergence[i] / maxDiv : 0;
  }

  spinUp() {}

  /** The flow never changes: nothing to upload. */
  advanceTo(t) { this.time = t; return false; }

  rmsSpeedCmYr() {
    let s2 = 0, a = 0;
    for (let i = 0; i < this.mesh.n; i++) {
      const v = this.velocity;
      s2 += (v[3 * i] ** 2 + v[3 * i + 1] ** 2 + v[3 * i + 2] ** 2) * this.mesh.area[i];
      a += this.mesh.area[i];
    }
    return Math.sqrt(s2 / a) / CM_YR_TO_RAD_MYR;
  }

  /** Exact flow at any point (same shape as Mantle.sample). */
  sample(x, hint = 0) {
    const { velocity, divergence } = kinematicFlow(x, this.options);
    return { velocity, buoyancy: divergence, node: hint };
  }

  getState() { return { kind: this.kind, options: this.options, time: this.time }; }

  setState(state) { this.time = state.time; }
}
