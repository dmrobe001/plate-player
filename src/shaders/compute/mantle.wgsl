// Mantle convection, caricatured as a Swift-Hohenberg pattern psi on the icosphere:
//   dpsi/dt = r psi - (lap + k0^2)^2 psi - psi^3 + F * blanket
// Crust velocity is then taken as -grad(psi) projected onto the sphere.
// Each substep is two dispatches: lap(psi), then lap(lap(psi)) + update.

@compute @workgroup_size(64)
fn mantle_laplacian(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.num_nodes) { return; }
    let c = grid[idx].psi;
    var s = 0.0;
    var n = 0u;
    for (var i = 0u; i < 6u; i++) {
        let a = adjacency[idx * 6u + i];
        if (a != NONE) { s += grid[a].psi; n += 1u; }
    }
    grid[idx].laplacian_psi = (s - f32(n) * c) * sim.laplacian_weight;
}

@compute @workgroup_size(64)
fn mantle_integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.num_nodes) { return; }
    let lap = grid[idx].laplacian_psi;
    var s = 0.0;
    var n = 0u;
    for (var i = 0u; i < 6u; i++) {
        let a = adjacency[idx * 6u + i];
        if (a != NONE) { s += grid[a].laplacian_psi; n += 1u; }
    }
    let bilap = (s - f32(n) * lap) * sim.laplacian_weight;
    let k2 = sim.mantle_wavenumber * sim.mantle_wavenumber;
    let p = grid[idx].psi;
    // (lap + k^2)^2 p = bilap + 2 k^2 lap + k^4 p
    let linear = sim.mantle_rayleigh * p - (bilap + 2.0 * k2 * lap + k2 * k2 * p);
    grid[idx].psi = p + sim.dt_mantle * (linear - p * p * p + sim.blanket_forcing * grid[idx].thermal_blanket);
}

@compute @workgroup_size(64)
fn mantle_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.num_nodes) { return; }
    var g = vec3f(0.0);
    let p = grid[idx].position.xyz;
    for (var i = 0u; i < 6u; i++) {
        let a = adjacency[idx * 6u + i];
        if (a != NONE) {
            let dir = grid[a].position.xyz - p;
            let d = length(dir);
            g += (dir / d) * ((grid[a].psi - grid[idx].psi) / d);
        }
    }
    // Project onto the tangent plane; flow runs down the psi gradient.
    grid[idx].velocity = vec4f(-(g - dot(g, p) * p), 0.0);
}
