// Area, volume and melt (design 4.8, phase E). CPU references:
// src/columns/area.js (the tessellation; it documents the construction) and
// src/columns/volume.js (thermal thinning, melt, mixing). This file follows
// them line by line.
//
//   columns_area         per surface column: its cell in the weighted Voronoi
//                        tessellation of the column positions (exact tiling of
//                        the sphere), from all columns in the 27 hash cells
//                        around it; area, centroid, major axis
//   columns_volume       per column: dilatation since last step thins the
//                        lithosphere (thermal age) and brings decompression
//                        melt; volume, composition, thickness = V / area
//   columns_init_volume  after loading: V = loaded thickness x cell area

const CLIP_SIDES: u32 = 12u;
const MAX_CELL_VERTICES: u32 = 32u;
const CLIP_LABEL: u32 = 0xFFFFFFFFu;
const HASH_LIST_CAP: i32 = 64;

fn excess_weight(r: f32) -> f32 {
    let r2 = r * r;
    return r2 * (0.5 + r2 * (5.0 / 24.0 + r2 * (61.0 / 720.0)));
}

// Tangent basis at unit vector n (tangentBasis in src/columns/flow.js).
fn tangent_e1(n: vec3f) -> vec3f {
    let ref_axis = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.9);
    return normalize(cross(ref_axis, n));
}

@compute @workgroup_size(64)
fn columns_area(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    particles[idx].cell_area_prev = particles[idx].cell_area;
    if (particles[idx].state != 1u) {
        // Slabs are not part of the surface tiling.
        particles[idx].cell_area = particles[idx].area_ref;
        particles[idx].cell_shape = vec4f(0.0);
        particles[idx].cell_clipped = 0u;
        particle_deltas[idx].cell_count = 0u;
        return;
    }
    let p = particles[idx].position.xyz;
    let e1 = tangent_e1(p);
    let e2 = cross(p, e1);
    let R = sim.cell_clip_radius;

    var poly: array<vec2f, 32>;
    var lab: array<u32, 32>;
    var n = CLIP_SIDES;
    for (var k = 0u; k < CLIP_SIDES; k++) {
        let t = 6.28318531 * f32(k) / f32(CLIP_SIDES);
        poly[k] = R * vec2f(cos(t), sin(t));
        lab[k] = CLIP_LABEL;
    }
    let bi = excess_weight(particles[idx].apothem);

    let home = hash_cell(p);
    let dim = i32(sim.hash_grid_dim);
    for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
        let nc = home + vec3i(x, y, z);
        if (any(nc < vec3i(0)) || any(nc >= vec3i(dim))) { continue; }
        var j = atomicLoad(&hash_grid[hash_index(nc)]);
        var iter = 0;
        while (j != NONE && iter < HASH_LIST_CAP && n >= 3u) {
            let next = particles[j].next_idx;
            iter++;
            if (j == idx || particles[j].is_active == 0u || particles[j].state != 1u) { j = next; continue; }
            // Half-plane c + nv.u <= 0 (halfPlane in area.js).
            let d = particles[j].position.xyz - p;
            let bj = excess_weight(particles[j].apothem);
            let aj = 1.0 + bj;
            let c = (bj - bi) - aj * dot(d, d) * 0.5;
            let nv = aj * vec2f(dot(d, e1), dot(d, e2));
            if (c + R * length(nv) <= 0.0) { j = next; continue; } // misses the clip disc
            // Sutherland-Hodgman (clipPolygon in area.js).
            var out: array<vec2f, 32>;
            var olab: array<u32, 32>;
            var m = 0u;
            for (var k = 0u; k < n; k++) {
                let cur = poly[k];
                let nxt = poly[(k + 1u) % n];
                let sc = c + dot(nv, cur);
                let sn = c + dot(nv, nxt);
                if (sc <= 0.0) {
                    if (m < MAX_CELL_VERTICES) { out[m] = cur; olab[m] = lab[k]; m++; }
                    if (sn > 0.0 && m < MAX_CELL_VERTICES) { out[m] = cur + (nxt - cur) * (sc / (sc - sn)); olab[m] = j; m++; }
                } else if (sn <= 0.0) {
                    if (m < MAX_CELL_VERTICES) { out[m] = cur + (nxt - cur) * (sc / (sc - sn)); olab[m] = lab[k]; m++; }
                }
            }
            poly = out;
            lab = olab;
            n = m;
            j = next;
        }
    }}}

    // Area (gnomonicArea), shape (polygonShape), clipped flag.
    var area = 0.0;
    var a2 = 0.0; var cx = 0.0; var cy = 0.0; var ixx = 0.0; var iyy = 0.0; var ixy = 0.0;
    var clipped = false;
    if (n >= 3u) {
        for (var k = 0u; k < n; k++) {
            let u = poly[k];
            let v = poly[(k + 1u) % n];
            let det = u.x * v.y - u.y * v.x;
            let lx = sqrt(1.0 + dot(u, u));
            let ly = sqrt(1.0 + dot(v, v));
            area += 2.0 * atan2(det, lx * ly + lx + ly + dot(u, v) + 1.0);
            a2 += det;
            cx += (u.x + v.x) * det;
            cy += (u.y + v.y) * det;
            ixx += (u.x * u.x + u.x * v.x + v.x * v.x) * det;
            iyy += (u.y * u.y + u.y * v.y + v.y * v.y) * det;
            ixy += (u.x * v.y + 2.0 * u.x * u.y + 2.0 * v.x * v.y + v.x * u.y) * det;
            if (lab[k] == CLIP_LABEL) { clipped = true; }
        }
    }
    if (clipped || n < 3u) { atomicAdd(&solver.clipped_cells, 1u); }
    particles[idx].cell_clipped = select(0u, 1u, clipped || n < 3u);
    var shape = vec4f(0.0);
    if (a2 > 0.0) {
        let A = 0.5 * a2;
        cx /= 3.0 * a2;
        cy /= 3.0 * a2;
        let cxx = ixx / 12.0 / A - cx * cx;
        let cyy = iyy / 12.0 / A - cy * cy;
        let cxy = ixy / 24.0 / A - cx * cy;
        let lambda = 0.5 * (cxx + cyy) + length(vec2f(0.5 * (cxx - cyy), cxy));
        shape = vec4f(cx, cy, 0.5 * atan2(2.0 * cxy, cxx - cyy), sqrt(max(lambda, 0.0)));
    }
    particles[idx].cell_shape = shape;
    // Tessellation neighbours and bending weights c = shared boundary / distance (flexure.wgsl).
    var nc = 0u;
    if (n >= 3u) {
        for (var k = 0u; k < n; k++) {
            let j = lab[k];
            if (j == CLIP_LABEL || nc >= 12u) { continue; }
            let len = length(poly[(k + 1u) % n] - poly[k]);
            particle_deltas[idx].cell_nbr[nc] = j;
            particle_deltas[idx].cell_coef[nc] = len / length(particles[j].position.xyz - p);
            nc++;
        }
    }
    particle_deltas[idx].cell_count = nc;
    // An empty cell (possible only with extreme weights) keeps its reference area.
    particles[idx].cell_area = select(particles[idx].area_ref, area, area > 0.0);
}

// volumeStep in src/columns/volume.js.
@compute @workgroup_size(64)
fn columns_volume(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    particles[idx].volume_added = 0.0;
    if (particles[idx].state != 1u) {
        particles[idx].thickness_km = particles[idx].crust_volume / particles[idx].area_ref;
        return;
    }
    let A = particles[idx].cell_area;
    let A0 = particles[idx].cell_area_prev;
    var V = particles[idx].crust_volume;
    if (A0 > 0.0) {
        var rho = particles[idx].composition_density;
        let ratio = A0 / A;
        let t = min(particles[idx].thermal_age_myr, thermal_age_cap_myr(rho)) * ratio * ratio;
        particles[idx].thermal_age_myr = t;
        let L = lithosphere_thickness_km(idx);
        let dA = A - A0;
        var added = 0.0;
        if (dA > 0.0) { added = sim.melt_fraction * sim.melt_depth_km * dA * max(1.0 - L / sim.melt_depth_km, 0.0); }
        if (added > 0.0) {
            let mixed = (rho * V + sim.melt_density * added) / (V + added);
            if (rho < 2850.0 && mixed >= 2850.0) {
                atomicAdd(&solver.breakups, 1u);
                particles[idx].age = 0.0;
            }
            rho = mixed;
            particles[idx].composition_density = rho;
            particles[idx].thermal_age_myr = min(t, thermal_age_cap_myr(rho));
            V += added;
            particles[idx].crust_volume = V;
            particles[idx].volume_added = added;
            add_budget(BUDGET_SOURCE, added);
        }
    }
    particles[idx].thickness_km = V / A;
}

// After loading (simulation.js): the loaded thickness defines the volume over
// the loaded geometry's cells; thermal ages are clamped to their caps.
@compute @workgroup_size(64)
fn columns_init_volume(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    particles[idx].crust_volume = particles[idx].thickness_km * particles[idx].cell_area;
    particles[idx].cell_area_prev = particles[idx].cell_area;
    particles[idx].thermal_age_myr = min(particles[idx].thermal_age_myr, thermal_age_cap_myr(particles[idx].composition_density));
}
