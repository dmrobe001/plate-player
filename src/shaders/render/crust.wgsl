// Crust particles as discs of radius = apothem, coloured by elevation.
// Bonds as stress glyphs:
//   - a line joining the two particle centres; black = compression,
//     white = tension, width proportional to |normal stress|
//   - a black tick across the bond midpoint, perpendicular to it, with
//     width proportional to shear stress. Its length is the shared edge of
//     two hexagons of the bonded particles' apothems, i.e. the "face" the
//     shear acts across.
// Width = mean apothem * |stress| / view.stress_ref_mpa, capped at one apothem.

// Hypsometric colour map; h is elevation above sea level in km.
fn elevation_color(h: f32) -> vec3f {
    if (h < 0.0) {
        let t = sqrt(clamp(-h / 6.0, 0.0, 1.0));
        return mix(vec3f(0.45, 0.72, 0.88), vec3f(0.03, 0.08, 0.28), t);
    }
    var c = mix(vec3f(0.20, 0.50, 0.22), vec3f(0.66, 0.66, 0.36), smoothstep(0.0, 1.5, h));
    c = mix(c, vec3f(0.50, 0.36, 0.24), smoothstep(1.5, 3.0, h));
    return mix(c, vec3f(0.95, 0.95, 0.97), smoothstep(3.0, 4.5, h));
}

// Orthonormal tangent vectors at unit normal n.
fn tangent_frame(n: vec3f) -> mat2x3f {
    let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t1 = normalize(cross(helper, n));
    return mat2x3f(t1, cross(n, t1));
}

const HIDDEN = vec4f(0.0); // degenerate clip position: the primitive is not drawn

struct DiscOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
}

@vertex
fn vs_disc(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> DiscOut {
    let p = particles[i];
    var out: DiscOut;
    if (p.is_active == 0u) { out.clip = HIDDEN; return out; }

    let n = p.position.xyz;
    let r = surface_radius(p);
    let uv = QUAD[v];
    let world = n * r + tangent_frame(n) * (uv * p.apothem * r);
    out.clip = view.view_proj * vec4f(world, 1.0);
    out.uv = uv;

    var color = elevation_color(p.elevation - sim.sea_level);
    if (p.state == 2u) { color *= 0.45; } // subducting slab: darkened
    let light = 0.55 + 0.45 * max(dot(n, view.light_dir.xyz), 0.0);
    out.color = color * light;
    return out;
}

@fragment
fn fs_disc(in: DiscOut) -> @location(0) vec4f {
    if (dot(in.uv, in.uv) > 1.0) { discard; }
    return vec4f(in.color, 1.0);
}

struct BondOut {
    @builtin(position) clip: vec4f,
    @location(0) @interpolate(flat) shade: f32,
}

// 12 vertices per bond: 0-5 the normal-stress line, 6-11 the shear tick.
@vertex
fn vs_bond(@builtin(vertex_index) v: u32, @builtin(instance_index) ci: u32) -> BondOut {
    let c = constraints[ci];
    var out: BondOut;
    out.clip = HIDDEN;
    out.shade = 0.0;
    if (c.is_active == 0u) { return out; }
    let pa = particles[c.particle_a];
    let pb = particles[c.particle_b];
    if (pa.is_active == 0u || pb.is_active == 0u) { return out; }

    let lift = 0.002; // draw just above the discs
    let a = pa.position.xyz * (surface_radius(pa) + lift);
    let b = pb.position.xyz * (surface_radius(pb) + lift);
    let len = length(b - a);
    if (len < 1e-6) { return out; }
    let along = (b - a) / len;
    let mid = 0.5 * (a + b);
    let across = normalize(cross(normalize(mid), along));

    let apothem = 0.5 * (pa.apothem + pb.apothem);
    let q = QUAD[v % 6u];
    var world: vec3f;
    if (v < 6u) {
        if ((view.flags & 1u) == 0u) { return out; }
        let half_width = 0.5 * min(apothem * abs(c.stress_normal_mpa) / view.stress_ref_mpa, apothem);
        world = mid + along * (q.x * 0.5 * len) + across * (q.y * half_width);
        out.shade = select(0.0, 1.0, c.stress_normal_mpa > 0.0);
    } else {
        if ((view.flags & 2u) == 0u) { return out; }
        let half_width = 0.5 * min(apothem * c.stress_shear_mpa / view.stress_ref_mpa, apothem);
        let half_length = 0.5 * (2.0 / sqrt(3.0)) * apothem; // half a hexagon edge
        world = mid * (1.0 + 0.0005 / length(mid)) + across * (q.x * half_length) + along * (q.y * half_width);
        out.shade = 0.0;
    }
    out.clip = view.view_proj * vec4f(world, 1.0);
    return out;
}

@fragment
fn fs_bond(in: BondOut) -> @location(0) vec4f {
    return vec4f(vec3f(in.shade), 1.0);
}
