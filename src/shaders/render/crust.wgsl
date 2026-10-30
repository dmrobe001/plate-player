// Crust particles as points and bonds as lines, coloured by damage.

struct PointOut {
    @builtin(position) clip: vec4f,
    @location(0) age: f32,
    @location(1) comp: f32,
    @location(2) elev: f32,
    @location(3) @interpolate(flat) state: u32,
}

@vertex
fn vs_point(@builtin(vertex_index) idx: u32) -> PointOut {
    let p = particles[idx];
    var out: PointOut;
    if (p.is_active == 0u) { out.clip = vec4f(0.0); return out; }
    out.clip = view.view_proj * vec4f(p.position.xyz * surface_radius(p), 1.0);
    out.age = p.age;
    out.comp = p.composition_density;
    out.elev = p.elevation;
    out.state = p.state;
    return out;
}

@fragment
fn fs_point(in: PointOut) -> @location(0) vec4f {
    var color = vec3f(0.2, 0.7, 0.3); // continental
    if (in.state == 2u) {
        color = mix(vec3f(0.8, 0.2, 0.0), vec3f(0.3, 0.0, 0.4), clamp(in.elev / -5.0, 0.0, 1.0)); // slab
    } else if (in.comp > 2800.0) {
        color = mix(vec3f(1.0, 0.3, 0.0), vec3f(0.1, 0.2, 0.4), clamp(in.age / 5.0, 0.0, 1.0)); // oceanic
    }
    return vec4f(color, 1.0);
}

struct LineOut {
    @builtin(position) clip: vec4f,
    @location(0) damage: f32,
}

@vertex
fn vs_line(@builtin(vertex_index) idx: u32) -> LineOut {
    let c = constraints[idx / 2u];
    var out: LineOut;
    if (c.is_active == 0u) { out.clip = vec4f(0.0); return out; }
    let p = particles[select(c.particle_a, c.particle_b, (idx % 2u) == 1u)];
    out.clip = view.view_proj * vec4f(p.position.xyz * (surface_radius(p) + 0.001), 1.0);
    out.damage = c.damage;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4f {
    return vec4f(mix(vec3f(0.1, 0.4, 0.8), vec3f(1.0, 0.1, 0.1), clamp(in.damage, 0.0, 1.0)), 1.0);
}
