// Crust particles as their reference shapes (phase G): an ellipse of mean
// radius = apothem, stretched by U = exp(L) in the body frame, coloured by
// elevation. Round columns are discs.
// Subducting slabs are white. Per-column glyphs (CPU reference
// src/columns/stress.js), from the forces the edges exert on the column:
//   - net force: one yellow arrow from the centre, the sum of the edge
//     forces (balanced by basal drag in this overdamped model), drawn as the
//     traction |F| / (h x 2 apothem), h the mean depth overlap of its edges;
//   - stress: the principal axes of the column's (Love-Weber) stress tensor,
//     each a pair of arrows from the centre, pointing out for tension (white)
//     and in for compression (black).
// Length = apothem * |stress| / view.stress_ref_mpa, capped at 2 apothems.

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

// Distinct colour per plate id: golden-angle hue steps (same as plateColor in
// src/plates/plates.js); grey for "not a plate".
fn plate_color(id: u32) -> vec3f {
    if (id == 0u) { return vec3f(0.35); }
    let h = (f32(id) * 137.508) % 360.0;
    let s = 0.55;
    let l = 0.55;
    let a = s * min(l, 1.0 - l);
    let k = (vec3f(0.0, 8.0, 4.0) + h / 30.0) % vec3f(12.0);
    return l - a * max(vec3f(-1.0), min(min(k - 3.0, 9.0 - k), vec3f(1.0)));
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
    // U uv = cosh(s) uv + sinh(s)/s L uv, L = [[sp, sq], [sq, -sp]], in (body_axis, n x body_axis).
    let s = length(vec2f(p.shape_p, p.shape_q));
    let f = select(1.0, sinh(s) / s, s > 1e-6);
    let su = cosh(s) * uv + f * vec2f(p.shape_p * uv.x + p.shape_q * uv.y, p.shape_q * uv.x - p.shape_p * uv.y);
    let b = p.body_axis.xyz;
    var frame = tangent_frame(n);
    if (dot(b, b) > 0.5) { frame = mat2x3f(b, cross(n, b)); }
    let world = n * r + frame * (su * p.apothem * r);
    out.clip = view.view_proj * vec4f(world, 1.0);
    out.uv = uv;

    var color = elevation_color(p.elevation - sim.sea_level);
    if ((view.flags & 4u) != 0u) { color = plate_color(plate_ids[i]) * mix(0.75, 1.1, smoothstep(-6.0, 4.0, p.elevation - sim.sea_level)); }
    if (p.state == 2u) { color = vec3f(1.0); } // subducting slab: white
    let light = 0.55 + 0.45 * max(dot(n, view.light_dir.xyz), 0.0);
    out.color = color * light;
    return out;
}

@fragment
fn fs_disc(in: DiscOut) -> @location(0) vec4f {
    if (dot(in.uv, in.uv) > 1.0) { discard; }
    return vec4f(in.color, 1.0);
}

struct GlyphOut {
    @builtin(position) clip: vec4f,
    @location(0) @interpolate(flat) color: vec3f,
}

// Net edge force (MPa km x unit length) and the stress tensor (MPa) of
// column i in its tangent frame; columnStress in src/columns/stress.js.
struct ColumnStress {
    force: vec2f,
    sxx: f32,
    syy: f32,
    sxy: f32,
    h: f32,
}

fn column_stress(i: u32, frame: mat2x3f) -> ColumnStress {
    var out: ColumnStress;
    let x = particles[i].position.xyz;
    let r = sim.shear_stiffness_ratio;
    let c = (3.0 + r) / (2.0 * sqrt(3.0) * (1.0 + r)); // normalStiffnessPerModulus
    var m = 0.0;
    let count = min(slot_words[i * SLOT_WORDS + SLOT_COUNT_WORD], SLOT_CAPACITY);
    for (var k = 0u; k < count; k++) {
        let e = edges[slot_words[i * SLOT_WORDS + SLOT_FIRST_WORD + k]];
        if (e.is_active == 0u || !(e.overlap_km > 0.0)) { continue; }
        let j = select(e.particle_a, e.particle_b, e.particle_a == i);
        let d = particles[j].position.xyz - x;
        let dist = length(d);
        if (dist < 1e-9) { continue; }
        let n2 = normalize(vec2f(dot(d, frame[0]), dot(d, frame[1])));
        let t2 = vec2f(-n2.y, n2.x);
        let S = e.stress_normal_mpa * e.overlap_km;
        let St = e.stress_shear_mpa * e.overlap_km;
        let f = c * e.rest * (S * n2 + St * t2); // (n and t reverse together at the b end)
        let lever = select(1.0 - e.lever_a, e.lever_a, e.particle_a == i);
        out.force += f;
        out.sxx += lever * dist * n2.x * f.x;
        out.syy += lever * dist * n2.y * f.y;
        out.sxy += 0.5 * lever * dist * (n2.x * f.y + n2.y * f.x);
        out.h += e.overlap_km;
        m += 1.0;
    }
    if (m == 0.0) { return out; }
    out.h /= m;
    let k = 1.0 / (particles[i].area_ref * out.h);
    out.sxx *= k;
    out.syy *= k;
    out.sxy *= k;
    return out;
}

// Vertex v (0-8) of an arrow along `dir` from the origin, `len` long, in
// (along, across) coordinates: a shaft quad (0-5) and a head triangle (6-8),
// pointing out (to len) or in (toward the origin, its base at len).
fn arrow_point(v: u32, len: f32, apothem: f32, inward: bool) -> vec2f {
    let head = min(0.4 * len, 0.45 * apothem);
    if (v < 6u) {
        let q = QUAD[v];
        let end = select(len - head, len, inward);
        return vec2f(0.5 * (q.x + 1.0) * end, q.y * 0.07 * apothem);
    }
    let w = min(0.22 * apothem, 0.6 * head);
    let base = select(len - head, len, inward);
    let tip = select(len, len - head, inward);
    if (v == 6u) { return vec2f(tip, 0.0); }
    return vec2f(base, select(-w, w, v == 8u));
}

fn glyph_frame(p: Particle) -> mat2x3f {
    let n = p.position.xyz;
    let b = p.body_axis.xyz;
    if (dot(b, b) > 0.5) { return mat2x3f(b, cross(n, b)); }
    return tangent_frame(n);
}

fn glyph_world(p: Particle, frame: mat2x3f, dir: vec2f, q: vec2f) -> vec3f {
    let along = frame * dir;
    let across = frame * vec2f(-dir.y, dir.x);
    let r = surface_radius(p) + 0.003; // just above the discs
    return p.position.xyz * r + (along * q.x + across * q.y) * r;
}

// 9 vertices per column: the net force arrow.
@vertex
fn vs_force(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> GlyphOut {
    var out: GlyphOut;
    out.clip = HIDDEN;
    out.color = vec3f(1.0, 0.82, 0.15);
    let p = particles[i];
    if (p.is_active == 0u) { return out; }
    let frame = glyph_frame(p);
    let s = column_stress(i, frame);
    let F = length(s.force);
    if (!(F > 0.0) || !(s.h > 0.0)) { return out; }
    let traction = F / (s.h * 2.0 * p.apothem);
    let len = min(p.apothem * traction / view.stress_ref_mpa, 2.0 * p.apothem);
    out.clip = view.view_proj * vec4f(glyph_world(p, frame, s.force / F, arrow_point(v, len, p.apothem, false)), 1.0);
    return out;
}

// 36 vertices per column: 2 principal axes x 2 directions x an arrow.
@vertex
fn vs_stress(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> GlyphOut {
    var out: GlyphOut;
    out.clip = HIDDEN;
    let p = particles[i];
    if (p.is_active == 0u) { return out; }
    let frame = glyph_frame(p);
    let s = column_stress(i, frame);
    if (!(s.h > 0.0)) { return out; }
    let mean = 0.5 * (s.sxx + s.syy);
    let R = length(vec2f(0.5 * (s.sxx - s.syy), s.sxy));
    let angle = 0.5 * atan2(2.0 * s.sxy, s.sxx - s.syy);
    let axis = v / 18u;          // 0: most tensile, 1: most compressive
    let sign = select(1.0, -1.0, (v / 9u) % 2u == 1u);
    let sigma = select(mean + R, mean - R, axis == 1u);
    let th = angle + f32(axis) * 1.5707963;
    let dir = sign * vec2f(cos(th), sin(th));
    let len = min(p.apothem * abs(sigma) / view.stress_ref_mpa, 2.0 * p.apothem);
    if (!(len > 0.0)) { return out; }
    out.color = select(vec3f(0.0), vec3f(1.0), sigma > 0.0);
    out.clip = view.view_proj * vec4f(glyph_world(p, frame, dir, arrow_point(v % 9u, len, p.apothem, sigma < 0.0)), 1.0);
    return out;
}

@fragment
fn fs_glyph(in: GlyphOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
