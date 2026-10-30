// Keep each particle's nearest mantle node up to date by walking downhill
// in distance over the icosphere's neighbour graph, starting from last
// step's node. On a Delaunay triangulation (which the icosphere is) this
// greedy walk ends at the true nearest vertex; particles move far less than
// one node spacing per step, so it usually takes zero or one hop.

const MAX_HOPS: u32 = 32u;

@compute @workgroup_size(64)
fn crust_locate(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= sim.max_particles || particles[idx].is_active == 0u) { return; }
    let p = particles[idx].position.xyz;
    var node = particles[idx].mantle_node;
    var best = distance(p, grid[node].position.xyz);
    for (var hop = 0u; hop < MAX_HOPS; hop++) {
        var next = node;
        for (var i = 0u; i < 6u; i++) {
            let a = adjacency[node * 6u + i];
            if (a == NONE) { continue; }
            let d = distance(p, grid[a].position.xyz);
            if (d < best) { best = d; next = a; }
        }
        if (next == node) { break; }
        node = next;
    }
    particles[idx].mantle_node = node;
}
