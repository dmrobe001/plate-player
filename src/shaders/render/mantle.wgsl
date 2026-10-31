// The mantle grid, coloured by buoyancy: orange = hot (upwelling),
// blue = cold (downwelling).

struct MantleOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) divergence: f32,
}

@vertex
fn vs_mantle(@builtin(vertex_index) idx: u32) -> MantleOut {
    var out: MantleOut;
    out.clip = view.view_proj * grid[idx].position;
    out.world = grid[idx].position.xyz;
    out.divergence = grid[idx].buoyancy * 1.5;
    return out;
}

@fragment
fn fs_mantle(in: MantleOut) -> @location(0) vec4f {
    let n = cross(dpdx(in.world), dpdy(in.world)); // flat-shaded facet normal
    let normal = select(normalize(in.world), normalize(n), length(n) > 0.0001);
    let diffuse = max(dot(normal, view.light_dir.xyz), 0.2);
    let t = clamp((in.divergence + 1.0) * 0.5, 0.0, 1.0);
    return vec4f(mix(vec3f(0.1, 0.2, 0.4), vec3f(0.8, 0.3, 0.1), t) * diffuse, 1.0);
}
