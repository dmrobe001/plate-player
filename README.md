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

**This is a work in progress, and so is its design.** The design document is a map,
not a contract. Don't pursue its goals too doggedly: when a step turns out to be
hard, costly, or a poor fit for what we've learned, stop and discuss a redesign
rather than forcing the plan through.

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
| **Colour by plate** | Each plate in its own colour; grey particles belong to no plate (fragments under 20 particles, or slabs) |

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

**Bonds** (`Constraint`). Each bond is a central spring with a rest length, a yield
stress, and damage in [0, 1]. Its stiffness comes from the lithosphere's Young's
modulus and elastic thickness, k = (√3/2)·E·hₑ (the 2D modulus of a triangular spring
network), with the physical E ≈ 70 GPa. Stress beyond an Arrhenius-weakened yield stress accumulates damage;
damage softens tension (never compression); heat slowly heals it.

**Motion** (`shaders/compute/solver.wgsl`). Overdamped extended position-based
dynamics (XPBD), with basal drag in the role of mass: each step minimises
Σ (γᵢ/2dt)|xᵢ − x̃ᵢ|² + U(x), where x̃ is where the mantle flow alone would carry
each particle, γ = area × asthenosphere viscosity / thickness, and U is the bond
energy. There is no inertia, and velocity is only a diagnostic. Bonds are solved by
Jacobi sweeps that keep the multipliers consistent with the displacement, accelerated
by Chebyshev extrapolation (32 iterations per step); collisions are position
projections found with a uniform hash grid. Yield and damage are applied once per step,
after the solve. With the default physical
values a single bond relaxes in ~40 years, so every step (16 kyr) is effectively
quasi-static: what limits rigidity at plate scale is solver convergence, not E (see
design.md, phase 2).

**Plates** (`src/plates/`). A plate is a connected component of intact bonds
(damage < 0.5) between surface particles, found exactly on the CPU every 0.25 Myr from a
readback. Plate ids persist from one labelling to the next (each component inherits the
id most of its particles had), so splits and merges show up as events. Each plate gets
an Euler pole fitted to its particles' displacement over the interval, plus its rms
speed and the rms of the non-rigid part.

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
│   ├── main.js                entry: device, initial state, UI, frame loop, window.plate API
│   ├── build-info.js          git commit injected at build time (written into records)
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
│   ├── plates/
│   │   ├── plates.js          plate labelling (union-find), persistent ids, Euler-pole fit
│   │   └── tracker.js         relabels during a run; plate series and event log for records
│   ├── record/
│   │   ├── recorder.js        record schedule, manifest, snapshots, ZIP export / resume
│   │   ├── sampling.js        particles -> fixed mesh (Gaussian kernel), geographic frame
│   │   └── zip.js             minimal stored-only ZIP reader/writer
│   ├── render/
│   │   ├── renderer.js        render pipelines and draw calls
│   │   └── camera.js          orbit camera
│   └── shaders/
│       ├── compute/           bindings.wgsl, locate, mantle, thermal, growth, solver
│       └── render/            common.wgsl, mantle, crust (discs + stress glyphs)
├── test/
│   ├── unit.test.mjs          CPU tests (Node only)
│   └── gpu.test.mjs           acceptance tests that drive the built page in Chromium
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

### Tests

```bash
npm run test:unit    # CPU code (zip, sampling, neighbour search, initial state); seconds
npm run test:gpu     # acceptance tests against dist/plate-player.html; minutes
npm test             # both, rebuilding first
```

The GPU tests drive the built page in Chromium through `window.plate` (the same
object you can poke at in the browser console). They need Playwright with Chromium
(`npx playwright install chromium`). By default the browser uses SwiftShader, a CPU
implementation of WebGPU, so results don't depend on your graphics card; set
`PLATE_GPU=hardware` to use the real GPU. Headless Chromium can't present a WebGPU
canvas, so the browser runs with a window; on a machine with no display (a server,
CI), run `xvfb-run -a npm run test:gpu`. Under SwiftShader a step takes ~0.4 s, so
the GPU suite takes several minutes.

Opening the page with `?paused` starts it paused, which scripts use to control
stepping exactly.

**What belongs in the test suite.** Keep tests cheap and about *mechanisms*:
invariants (every slot free or active exactly once), small analytic cases (two
particles relaxing on a known time scale), file formats. Don't accumulate regression
tests on high-level results (plate speeds, land fraction, how a 100 Myr run looks).
They need lots of compute and pin down behaviour that is supposed to keep changing.
Measure those once when a phase lands, write the number in `docs/design.md`, and
move on.

## Recording

Every run is recorded unless **Record this run** is unchecked (design.md §8.3 has
the full rationale):

- **Field series** every *Fields every* Myr (default 1): particle quantities averaged
  onto a fixed icosphere mesh (10 242 nodes, ~220 km spacing) with a Gaussian kernel
  one particle-spacing wide. Fields: surface height, crust thickness, density, age,
  temperature, east/north velocity (mm/yr), bond damage, slab fraction, and
  `support` (~1 under continuous crust, →0 in gaps; NaN values mark "no crust here").
- **Snapshots** every *Snapshot every* Myr (default 25), plus one at the start and on
  **Snapshot now**: the raw GPU buffers, enough to resume the run exactly.
- **Download record** saves everything as a ZIP; **Resume from record** loads the
  latest snapshot in a ZIP and continues from it (as a new record that notes where it
  came from).

Inside the ZIP, `manifest.json` describes everything else: the seed, initial-condition
options, every parameter, the git commit of the build, the memory layout of each
snapshot buffer, and the shape, dtype and units of every array. Arrays are raw
little-endian binaries. Reading a record in Python:

```python
import json, zipfile, numpy as np
z = zipfile.ZipFile("plate-player_seed1_step600.zip")
m = json.loads(z.read("manifest.json"))
def array(entry):
    return np.frombuffer(z.read(entry["file"]), dtype=entry["dtype"]).reshape(entry["shape"])
lat_lon = array(m["mesh"]["files"]["lat_lon"])            # (nodes, 2) degrees
t = array(m["fields"]["time"])                            # (T,) Myr
fields = {v["name"]: array(v) for v in m["fields"]["variables"]}
height = fields["surface_height"]                         # (T, nodes) km above sea level
```

Recording costs ~0.4 MB per field sample and ~7 MB per snapshot. A readback pauses
the simulation for a few milliseconds, so each sample is taken at an exact step.

## Reproducibility

What we want: a seed reproduces the initial world, a record says exactly what made it,
and a snapshot resumes a run. What we don't chase: **bit-identical GPU results**. On
GPUs that is more trouble than it's worth, since thread timing, driver versions and
floating-point contraction all vary. A resumed or re-run simulation should match
*statistically and visually*, not byte for byte.

- The initial state is a pure function of the seed and `INIT_DEFAULTS` (CPU code).
- A snapshot restores the complete state, so resuming continues the same run. The
  test checks that the continuation matches to within a small tolerance.
- Known sources of run-to-run variation are listed in each record's manifest (which
  free slot a splitting particle receives; neighbour-list order in the collision
  search). We note them; we don't engineer them away.
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
- [ ] **Slabs melt within a few frames.** Slab heating relaxes at 2.5 /Myr (now
      integrated exactly) and thickness loss is 0.1 km per K, so a 7 km slab is
      gone in about two steps. The rates (`slab_heating_rate`,
      `slab_assimilation_km_per_k`) should be much slower so slabs persist long
      enough to pull; design phase 7 replaces this mechanism.
- [ ] **No ocean depth–age relation.** Real sea floor deepens about 3 km from ridge
      to old basin (depth ∝ √age, from cooling of the lithospheric mantle). Here
      only the 7 km crust contracts, about 0.25 km. Adding a lithosphere term
      ∝ √(κ·age) to elevation would give ridges and basins.
- [x] ~~Bond "shear" resisted rigid rotation~~ (it was a velocity difference, not a
      strain). Removed in phase 2: bonds are central springs. A rotation-free shear
      measure comes with the per-particle strain fit (design phase 4); until then the
      shear ticks are blank.
- [x] ~~Young's modulus had no dynamical effect.~~ Phase 2: bond stiffness and drag
      are physical, and the two-particle relaxation matches τ = γ/2k (tested).
- [x] ~~Plate-scale stress didn't converge with plain Jacobi~~ (median 56 MPa of
      solver residual at 70 GPa, spurious yield). Chebyshev acceleration with 32
      iterations brings it to ~3 MPa with no spurious yield (design.md, Chebyshev
      findings). The plate level (design phase 5) is next.
- [ ] **Long steps can trip the browser's GPU watchdog.** One step is one GPU
      submission; with many iterations on a slow GPU (128 iterations under the
      SwiftShader software renderer) it exceeds Chromium's time limit and the device
      is lost. Split long steps into several submissions if this bites on real GPUs.
- [ ] **Growth recreates bond/contact conflicts.** The initial state is now
      stress-free (apothems are shrunk so no pair overlaps; with the mantle off, the
      stress is exactly zero), but a growing particle's bonds keep their rest length
      while collisions push to the new apothem sum. Design phase 3 stops bonded pairs
      from colliding.
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
- [ ] **The mantle pattern takes hundreds of Myr to emerge.** Pattern time advances
      at 0.0156 units per Myr and the growth rate is r = 0.2, so one e-folding of
      the fastest mode takes ~320 Myr. Meanwhile off-wavelength noise decays (max
      |ψ| falls from 0.05 to 0.009 in the first 1.6 Myr) and plates drift at
      ~1–2 cm/yr. (Inherited from the Phase 5 constants.) Either start ψ from a
      developed pattern or raise `mantle_time_per_myr`.
- [ ] **Plates barely move for the first hundreds of Myr** (rms ~0.5 mm/yr, because of
      the slow mantle spin-up above), so plate kinematics and rigidity can't be judged
      yet. Note also that a developed pattern (|ψ| ~ 0.5, k₀ = 4) times
      `sh_grad_to_cm_yr` = 25 would give ~50 cm/yr, so the spin-up and the velocity
      scale should be fixed together.
- [ ] **Mantle has no toroidal (rotational) flow.** Velocity = −∇ψ is curl-free, so
      transform-style motion arises only from crust interactions.
- [x] ~~Mantle time had no stated relation to Myr, and the explicit step size was
      hand-tuned.~~ Pattern time now advances at `mantle_time_per_myr`, and the
      substep is derived from a power-iteration estimate of the operator's largest
      eigenvalue (half the explicit-Euler limit). For the record: the old step sat at
      λ_max·dt ≈ 0.75 against a limit of 2, not the ≈ 1.5 this list first claimed.

**Correctness / GPU**
- [x] ~~Race in `crust_solve_collisions` on the subduction state flip.~~ The flip is
      written to `next_state` and applied in `crust_apply_deltas`.
- [x] ~~Race in free-list pops.~~ Splits now reserve slots with tickets and a
      single-thread `crust_commit_splits` pops them; counters never wrap. The GPU
      tests check that every slot is either active or free, exactly once.
- [x] ~~Explicit relaxation rates (velocity drag, temperatures) had dt-dependent
      stability.~~ They are integrated as exact exponentials, and area growth is a
      rate per Myr. A 10× time step runs stably (tested).
- [ ] Healing is effectively off. At surface temperatures exp(−Eₐ/RT) underflows
      the `> −50` guard, so only near-mantle-temperature bonds heal.

**Performance**
- [x] ~~`nearest_node` brute force over all mantle nodes.~~ Each particle caches
      its node and updates it by a greedy walk over the icosphere (`locate.wgsl`).
- [ ] Every dispatch covers the full capacity (25 000 particles, 100 000 bonds)
      regardless of how many are active. An indirect dispatch on the live count
      would help.
- [ ] Collision hash cells (3/32 ≈ 0.094) are about 2.7× the contact distance;
      cells ≈ contact distance would scan about 10× fewer candidates.
- [ ] Decouple simulation steps from display frames (fixed steps per frame, or a
      step budget) so speed doesn't depend on the monitor.

**For game use**
- [ ] Export: rasterise a record's `surface_height` to an equirectangular height
      map and save as PNG (the record already has the data on a mesh).
- [ ] Expose `INIT_DEFAULTS` (continent fraction, plate count, etc.) in the UI.
