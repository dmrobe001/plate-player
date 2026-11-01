// Bindings and helpers shared by all render entry points.

struct View {
    view_proj: mat4x4f,
    light_dir: vec4f,
    stress_ref_mpa: f32,   // stress at which a stress line is one apothem wide
    flags: u32,            // bit 0: normal-stress lines, bit 1: shear ticks, bit 2: colour by plate
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> edges: array<Edge>;
@group(0) @binding(2) var<storage, read> grid: array<GridNode>;
@group(0) @binding(3) var<storage, read> plate_ids: array<u32>;   // from PlateTracker
@group(1) @binding(0) var<uniform> view: View;
@group(1) @binding(1) var<uniform> sim: SimParams;

// Two triangles covering the square [-1, 1]^2.
const QUAD = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

// Particles are drawn slightly above the mantle sphere, raised or lowered by
// elevation (relative to sea level) so that slabs tuck under overriding crust.
fn surface_radius(p: Particle) -> f32 {
    return 1.02 + clamp((p.elevation - sim.sea_level) * 0.002, -0.01, 0.02);
}
