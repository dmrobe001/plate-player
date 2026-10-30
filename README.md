# Plate Player

A WebGPU particle model of plate tectonics, meant for generating worlds for a game.
The crust is a sheet of bonded particles on a unit sphere, carried by a mantle flow
field. It grows where the flow diverges, breaks where it is overstressed, and
subducts where it converges. Everything runs on the GPU in one HTML page.

The build produces a single self-contained file, `dist/plate-player.html`. Open it
in a WebGPU-capable browser (recent Chrome or Edge; Firefox and Safari have partial
support). It works straight from disk, with no server needed.

**Where this is going:** [`docs/design.md`](docs/design.md) is the plan for the next
version: overdamped implicit solver, unified bond/contact edges, explicit slabs, plates as
Euler-pole rotations, and the record/forcing contracts for nesting finer models below
this one. The to-do list at the end of this README is the problem inventory that plan
answers.

## Quick start

```bash
npm install        # once: installs esbuild, the only dependency
npm run build      # writes dist/plate-player.html
npm run watch      # rebuilds on every save under src/; reload the browser to see changes
```

On the page, drag to rotate, scroll to zoom. **Seed** + **Regenerate** rebuilds the
initial state; the same seed always gives the same initial world. You can also put
the seed in the URL: `plate-player.html?seed=42`.

## What you are looking at

| Glyph | Meaning |
|---|---|
| Disc | One crust particle. Radius = the particle's apothem; colour = elevation relative to sea level (blues below, greens/browns/white above). Darkened discs are subducting slabs. |
| Black line between centres | Bond in **compression**. Width ∝ \|normal stress\| |
| White line between centres | Bond in **tension**. Width ∝ normal stress |
| Black tick across a bond's midpoint | **Shear** stress on the bond. Width ∝ shear stress; length = one hexagon edge (the "face" between the two particles) |
| Background sphere | Mantle grid, coloured by flow divergence: orange = upwelling, blue = downwelling |

Line width = mean apothem × |σ| / σ_ref, capped at one apothem. σ_ref is the
**Line scale** slider (log scale, default ≈ the 150 MPa yield stress).

## Model in one page

Units: lengths are on the unit sphere (1 = Earth radius, 6371 km) unless suffixed
`_km`; time in Myr (millions of years); temperature in K; density in kg/m³.

**Mantle** (`shaders/compute/mantle.wgsl`). A scalar field ψ on the 642 vertices of
a subdivided icosahedron evolves by a Swift–Hohenberg equation,
∂ψ/∂t = rψ − (∇² + k₀²)²ψ − ψ³ + F·blanket. This is a pattern-forming
equation that settles into cells of preferred wavelength 2π/k₀, standing in for
convection cells. The crust's surface velocity target is −∇ψ, projected onto
the sphere. The "thermal blanket" term lets thick continental crust warm the
mantle beneath it and slabs cool it.

**Crust particles** (`Particle` in `src/gpu/layout.js`). Each one is a hexagonal
column with thickness, composition density, temperature, age, and area
(2√3·apothem²). Its density follows thermal expansion and its elevation follows
Airy isostasy: thickness × (1 − ρ/ρ_mantle), with extra depression under water.

**Bonds** (`Constraint`). Each bond has a rest length, a yield stress, and damage
in [0, 1]. Stress beyond an Arrhenius-weakened yield stress accumulates damage;
damage weakens tension and shear (never compression); heat slowly heals it.

**Motion** (`shaders/compute/solver.wgsl`). Position-based dynamics (PBD): each
frame, velocities relax toward the mantle flow, positions are predicted, then three
Jacobi iterations of bond constraints and disc-disc collisions correct them.
Collisions use a uniform hash grid.

**Growth and subduction** (`growth.wgsl`, `solver.wgsl`, `thermal.wgsl`). Particles
over strong divergence grow in area. Past 3× the base area they split, and the
child is fresh oceanic crust. When two surface discs overlap deeply, the lower one
becomes a slab (state 2): it slides under with only basal friction, heats up, and
thins until it melts. Its index then returns to a free list for reuse.

### Initial conditions (`src/init/initial-state.js`)

1. N points scattered uniformly on the sphere, where N is chosen so the hexagons
   would tile 85% of it.
2. 100 sweeps of pure pairwise repulsion (Gauss–Seidel) push discs apart. The
   largest remaining overlap is shown in the panel; typical overlaps are ≪1%.
3. Continental vs oceanic crust comes from thresholding fractal (fBm) Perlin
   noise at the quantile that makes 35% of particles continental. Continental
   thickness rises from 30 km at the coast to 45 km in the interior.
4. Each particle bonds to its nearest neighbours within 1.3 contact distances
   (closest pairs first, at most 6 bonds each). Rest length is the current
   distance, so the bond lattice starts stress-free.
5. Bond damage comes from domain-warped Voronoi (Worley) noise around 12 random
   plate centres. Bonds near cell borders are damaged, which seeds a network of
   weak plate boundaries with triple junctions.
6. Velocities start at zero, and ψ starts as small random noise.

All parameters are in `INIT_DEFAULTS` and `CRUST` at the top of that file.

## Project layout

```
plate-player/
├── build.mjs                  bundle + inline → dist/plate-player.html
├── package.json               npm scripts; esbuild is the only dependency
├── docs/
│   └── design.md              design plan for the next version
├── src/
│   ├── index.html             page template (build inlines CSS + JS where the markers are)
│   ├── styles.css
│   ├── main.js                entry: device, initial state, UI, frame loop
│   ├── gpu/
│   │   ├── device.js          WebGPU init, shader compile-error reporting
│   │   └── layout.js          GPU struct definitions (Particle, Constraint, GridNode)
│   ├── geometry/
│   │   ├── icosphere.js       mantle grid mesh + adjacency
│   │   └── vec.js             vector / matrix helpers
│   ├── init/
│   │   ├── initial-state.js   the initial-condition recipe above
│   │   ├── noise.js           seeded Perlin noise, fBm, quantiles
│   │   ├── cell-list.js       CPU neighbour search
│   │   └── random.js          seeded RNG, uniform points on the sphere
│   ├── sim/
│   │   ├── params.js          every simulation parameter, with units
│   │   └── simulation.js      GPU buffers, compute pipelines, per-frame schedule
│   ├── render/
│   │   ├── renderer.js        render pipelines and draw calls
│   │   └── camera.js          orbit camera
│   └── shaders/
│       ├── compute/           bindings.wgsl, mantle, thermal, growth, solver
│       └── render/            common.wgsl, mantle, crust (discs + stress glyphs)
└── dist/                      build output (git-ignored)
```

### Design rules worth keeping

- **One source of truth for memory layouts.** GPU structs (`src/gpu/layout.js`) and
  the `SimParams` uniform (`src/sim/params.js`) are declared once in JavaScript as
  lists of fields, and the build generates the matching WGSL `struct` text from them.
  To add a particle property or a parameter, add one line there. Never hand-count
  byte offsets.
- **Shaders are plain `.wgsl` files.** esbuild imports them as strings. Each
  pipeline's code is the generated structs, followed by a shared `bindings.wgsl` or
  `common.wgsl`, followed by the feature files, all concatenated.
- **Seeded randomness only.** Use `mulberry32(seed)`, never `Math.random()`, so a
  seed reproduces a world.

## Workflow

1. Install [Node.js](https://nodejs.org) (version 18 or newer), then run
   `npm install` in this folder.
2. Run `npm run watch` and open `dist/plate-player.html` in Chrome.
3. Edit anything under `src/`, save, and reload the page. Shader compile errors
   appear in the panel with the offending line.
4. Commit source changes only; `dist/` is rebuilt, not versioned.

To **add a simulation parameter**, add a line to `SIM_PARAMS` in `params.js` and
use it as `sim.<name>` in any shader. To **add a compute stage**, write an
`@compute` function in one of the `shaders/compute/*.wgsl` files and add its name
to `entryPoints` and to the schedule in `encodeStep` (`simulation.js`).

### Testing without a browser window

The initial-condition code is plain JavaScript and runs in Node:

```bash
node -e "import('./src/init/initial-state.js').then(m => console.log(m.generateInitialState(642, { seed: 1 }).stats))"
```

For the GPU part, headless Chromium's software WebGPU (SwiftShader) can run the
page but cannot present the canvas. Use a headed browser under a virtual display
(`xvfb-run`) to take screenshots with Playwright. It runs at about one step per
second, which is enough to check that shaders compile and things look right.

## Reproducibility

- The initial state is a pure function of the seed and `INIT_DEFAULTS`.
- GPU evolution is *mostly* deterministic. Position corrections are summed with
  integer atomics, which is order-independent. Free-list pops (particle
  splitting) and the subduction state flip depend on thread scheduling, so two
  runs can diverge once particles split or subduct.
- The simulation advances one step per displayed frame, so simulated time per
  wall-clock second depends on the monitor's refresh rate and GPU speed. The
  panel reports steps and simulated Myr.

## To do

Ordered roughly by how much each item affects the plausibility of generated worlds.

**Physics**
- [ ] **Slab pull never fires.** It requires slab density > 3300, but slabs are
      3000 kg/m³ crust being *heated*, so their density stays below about 3120.
      The fix is to include the cold lithospheric mantle that moves with the slab
      (density ≈ 3300 × (1 + α ΔT)). The pull magnitude (`× 0.02` rad/Myr per
      kg/m³) is also orders of magnitude above plate speeds. Its direction
      follows the mantle flow (biased by `+vec3(0.001)`), not the trench normal.
- [ ] **Slabs melt within a few frames.** Each step adds 2.5 × dt ≈ 4% of the
      temperature gap, and the thickness loss is 0.1 km per K, so a 7 km slab is
      gone in about two steps. The rate should be per Myr and much slower, so
      slabs persist long enough to pull.
- [ ] **No ocean depth–age relation.** Real sea floor deepens about 3 km from ridge
      to old basin (depth ∝ √age, from cooling of the lithospheric mantle). Here
      only the 7 km crust contracts, about 0.25 km. Adding a lithosphere term
      ∝ √(κ·age) to elevation would give ridges and basins.
- [ ] **Bond "shear" resists rigid rotation.** The tangential error is the relative
      displacement *this step*, which is a velocity difference, not a strain. A
      rigidly rotating plate therefore shears every bond, and the term acts as
      drag. Options: give particles an orientation, or measure shear from the
      change in bond angle relative to neighbouring bonds.
- [ ] **Young's modulus has no dynamical effect.** PBD with stiffness 1 makes bonds
      as stiff as the iteration count allows; `crust_youngs_modulus_gpa` only
      converts strain to stress for yielding and display, and `compliance` is
      unused. Using XPBD (extended PBD, where compliance = 1/stiffness) would make
      stiffness a real material parameter independent of iteration count.
- [ ] Shear stress uses Young's modulus E rather than the shear modulus
      G = E / (2(1+ν)).
- [ ] **Bond rest length vs. contact distance.** Bonds hold particles at their rest
      length while collisions push any pair closer than the sum of apothems apart,
      so the two rules fight wherever rest < contact. Growth makes this worse,
      because a growing particle's bonds keep their old rest length. Consider
      excluding bonded pairs from collision handling, or keeping rest length
      = apothem sum.
- [ ] **Bonds never form and never die from damage.** A fully damaged bond still
      resists compression and keeps 10% of its shear stiffness, and colliding
      plates never weld (no suturing or accretion).
- [ ] **Water loading factor.** A depth d below sea level should become
      d·ρ_m/(ρ_m − ρ_w) ≈ 1.43 d; the code gives d·(1 + ρ_w/ρ_m) ≈ 1.30 d.
- [ ] **Mass is not conserved at splits.** The parent keeps its thickness at half
      the area, and the child is reset to 7 km oceanic crust, so continental
      material vanishes when a continent rifts.
- [ ] **Subduction choice.** The lower disc of any deeply overlapping pair
      subducts, including continent under continent. The polarity is decided per
      pair, so a trench can flip polarity along its length.
- [ ] **Mantle has no toroidal (rotational) flow.** Velocity = −∇ψ is curl-free, so
      transform-style motion arises only from crust interactions.
- [ ] Mantle time is a pattern-time variable (50 × 5×10⁻⁶ per frame) with no stated
      relation to Myr, and the explicit Euler step is near its stability limit
      for the ∇⁴ term (λ_max·dt ≈ 1.5 of the limit 2). Changing
      `laplacian_weight` or the subdivision level can make ψ blow up.

**Correctness / GPU**
- [ ] Race in `crust_solve_collisions`: a thread writes `particles[idx].state = 2`
      while neighbours read `state` in the same dispatch. Record the flip in a
      separate buffer and apply it afterwards.
- [ ] Race in free-list pops: `atomicSub` on an empty counter wraps to 0xFFFFFFFF
      before being restored, and concurrent threads can read that value.
- [ ] Healing is effectively off. At surface temperatures exp(−Eₐ/RT) underflows
      the `> −50` guard, so only near-mantle-temperature bonds heal.

**Performance**
- [ ] `nearest_node` is a brute-force loop over all 642 mantle nodes, run three
      times per particle per frame (thermal, accumulation, predict). Caching the
      node index per particle (search only the previous node's neighbours) removes
      about 50 M distance evaluations per frame.
- [ ] Every dispatch covers the full capacity (25 000 particles, 100 000 bonds)
      regardless of how many are active. An indirect dispatch on the live count
      would help.
- [ ] Collision hash cells (3/32 ≈ 0.094) are about 2.7× the contact distance;
      cells ≈ contact distance would scan about 10× fewer candidates.
- [ ] Decouple simulation steps from display frames (fixed steps per frame, or a
      step budget) so speed doesn't depend on the monitor.

**For game use**
- [ ] Export: read back particle positions and elevations, rasterise them to an
      equirectangular height map, and save as PNG.
- [ ] Expose `INIT_DEFAULTS` (continent fraction, plate count, etc.) in the UI.
