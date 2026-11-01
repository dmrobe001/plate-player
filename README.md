# Plate Player

A WebGPU particle model of plate tectonics, meant for generating worlds for a game.
The crust is a sheet of bonded particles on a unit sphere, carried by a mantle flow
field. It grows where the flow diverges, breaks where it is overstressed, and
subducts where it converges. Everything runs on the GPU in one HTML page.

The build produces a single self-contained file, `dist/plate-player.html`. Open it
in a WebGPU-capable browser (recent Chrome or Edge; Firefox and Safari have partial
support). It works straight from disk, with no server needed.

**Where this is going:** [`docs/design.md`](docs/design.md) is the plan for the next
version: one pair-interaction law (no bonds vs. contacts; graded, history-dependent
integrity), exact area and volume budgets, explicit slabs, and the record/forcing contracts for nesting finer models below
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
| Black line between centres | Edge in **compression**. Width ∝ \|normal stress\| |
| White line between centres | Edge in **tension** (only edges with integrity > 0 carry tension). Width ∝ normal stress |
| Black tick across an edge's midpoint | **Shear** stress on the edge. Width ∝ shear stress; length = one hexagon edge (the "face" between the two particles) |
| Background sphere | Mantle grid, coloured by flow divergence: orange = upwelling, blue = downwelling |
| **Colour by plate** | Each plate in its own colour; grey particles belong to no plate (fragments under 20 particles, or slabs) |

Line width = mean apothem × |σ| / σ_ref, capped at one apothem. σ_ref is the
**Line scale** slider (log scale, default ≈ the 150 MPa yield stress).

## Model in one page

Units: lengths are on the unit sphere (1 = Earth radius, 6371 km) unless suffixed
`_km`; time in Myr (millions of years); temperature in K; density in kg/m³.

**Mantle** (`src/mantle/`). A cheap, living stand-in for convection, run on the CPU
on a 2562-node icosphere. A buoyancy field (the mid-mantle temperature anomaly) drives a
poloidal flow beneath the plates: hot regions upwell and diverge, cold regions converge.
The field is advected by its own flow, decays with a plume lifetime (~120 Myr), and is
replenished by randomly born hot plumes and cold drips (~0.3 per Myr each, so ~35 are
active, like Earth's ~40 hotspots). The pattern therefore never settles; it decorrelates
over ~100 Myr, like a lava lamp. The flow from an anomaly is screened over the mantle's
depth (2900 km). The velocity scale is set so the long-run rms speed is ~4 cm/yr,
about Earth's rms plate speed. The mantle runs alone for 400 Myr before the crust
starts. It does not yet feel the crust (no blanketing under continents, no slabs as
cold sinks). Details and grounding: `docs/design.md`, "Mantle and basal drag".

**Basal drag** (`inverse_drag` in `solver.wgsl`). Each particle feels Couette shear
across the asthenosphere: traction per area = η·(v_mantle − v_plate)/h. Here η is the
asthenosphere viscosity, and h is the channel from the plate's base (oceanic: 2.32√(κ·age),
capped at 100 km; continental: 150 km) down to the base of the low-viscosity layer
(300 km). Deep keels drag hard and young ocean drags weakly. A plate's speed is not
imposed: it comes out of the force balance between this drag, the bonds holding the
plate together, collisions with other plates, and slab pull through slab edges (ridge
push exists but is off until calibration).
η is the effective drag coefficient to tune.

**Crust particles** (`Particle` in `src/gpu/layout.js`). Each one is a hexagonal
column with thickness, composition density, temperature, age, and area
(2√3·apothem²). Its elevation (the top of the column) is a free coordinate that
relaxes toward the Airy height from its column profile (below), with the viscous
rebound time of a column-sized load (~30 kyr), while bending couples neighbours: a
thin elastic plate of rigidity D = E hₑ³/12(1 − ν²) over the column's elastic
thickness, discretised on the tessellation (`shaders/compute/flexure.wgsl`; reference
`src/columns/flexure.js`; matches the line-load flexure profile to 1.4%). It is not
part of `position`, which stays on the unit sphere.

**Column profiles** (`ColumnProfile`, `shaders/compute/profile.wgsl`; CPU reference
`src/columns/profile.js`). Every step, each column's vertical structure is computed
from its state on 8 depth slices (3 in the crust, 5 in the lithospheric mantle):
- **temperature**: oceanic columns cool as a half-space from their age (capped at
  ~59 Myr, where the lithosphere reaches 100 km, so old sea floor flattens);
  continents have a steady geotherm with radiogenic heat in the upper 15 km;
- **density** from composition and temperature, and **lithostatic pressure**;
- **ductile strength and viscosity** from laboratory creep laws (quartzite for
  continental crust, diabase for oceanic crust, olivine for mantle) at a reference
  strain rate of 10⁻¹⁵/s;
- column totals: lithosphere base, **Airy height** of the whole column (crust plus
  cooled mantle, water-loaded below sea level), **gravitational potential energy**
  (GPE), surface heat flow, Moho temperature, and the brittle–ductile transition
  (where Byerlee friction and creep cross).

So sea floor deepens with √age (ridge 2.9 km deep, old basins 5.6 km), and GPE
differences give the ridge push of half-space cooling. Isostasy and the edge law read
the profile, and so does the GPE push (below). Lithosphere thickness comes from a
thermal age for continents too (capped at 150 km), so stretching can thin it.

**Edges** (`Edge`, `shaders/compute/edges.wgsl`). There is one kind of edge: the
interface between two neighbouring columns. An edge exists exactly while its pair is
in range: a map-view gap below 10% of the apothem sum *and* overlapping depth
intervals. It is created when the pair comes into range and dropped when it leaves,
and in between it keeps its state across steps. **Integrity** in [0, 1] replaces the
old bond/contact split. Initial bonds start at integrity 1 − damage; edges made later
start at 0, except the parent–child edge of a split, which starts at 1. Each particle
also has a slot list of its edges, sorted by angle, rebuilt every step.

**Edge law** (`shaders/compute/law.wgsl`; CPU reference `src/edges/law.js`). An edge
couples its two columns over the depth range they share, in 8 slices. Each slice uses
the two columns' temperature and creep strength at that depth (from the column
profiles). Each slice is a spring (Young's modulus 70 GPa) in series with a dashpot
(the pair's creep viscosity), and keeps its own stress from step to step. Cold slices
act as springs and hot ones as dashpots, so the elastic thickness emerges: 47 km for a
pair of 100 Myr old ocean. Each slice's stress is capped by the yield-strength
envelope:
- compression: the weaker of Byerlee friction (plus cohesion × integrity) and creep;
- tension: cohesion × integrity where the slice is brittle, or the creep flow stress
  where it is ductile.

So an edge at integrity 0 between cold columns only pushes (a contact), and its
brittle slices crack in tension and close exactly at touching. Within a step the law is
linear, with the total force clamped to the depth-integrated caps. Only the normal
(push–pull) part exists: tangential friction needs a per-particle orientation first.

**Irreversible step** (`law.wgsl`, `shaders/compute/flow.wgsl`; references
`src/edges/law.js`, `src/columns/flow.js`). After each solve:
- **Inelastic strain.** Each edge records the strain its stiffest slice could not
  store elastically.
- **Volumetric flow.** Each column changes its reference area by twice the mean of
  its edges' inelastic strain; pure shear leaves areas alone. The rest length (sum of
  apothems) follows the area. Edges remember the pair's distance, not its strain, so
  that change adds no stress.
- **Integrity.** Edges lose integrity in proportion to the plastic work done on them,
  and heal toward 1 with a thermally activated time: 0.1 Myr at mantle temperature,
  ~50 Myr at 1200 K, never in cold lithosphere.
- **GPE push** (ridge push and plateau collapse): columns are pushed down the
  gradient of gravitational potential energy. It is implemented but **off by
  default** (`gpe_force_factor` = 0): against today's basal drag it pushes margin
  columns at metres per year, so it waits for calibration.


**Motion** (`shaders/compute/solver.wgsl`). Overdamped extended position-based
dynamics (XPBD), with basal drag in the role of mass: each step minimises
Σ (γᵢ/2dt)|xᵢ − x̃ᵢ|² + U(x), where x̃ is where the mantle flow alone would carry
each particle, γ = area × asthenosphere viscosity / thickness, and U is the edge
energy. There is no inertia, and velocity is only a diagnostic. All edges are solved
by Jacobi sweeps that keep the multipliers consistent with the displacement,
accelerated by Chebyshev extrapolation (32 iterations per step); pairs coming into
range are found once per step with a uniform hash grid. Yield and integrity loss are
applied once per step, after the solve. With the default physical
values a single edge relaxes in ~40 years, so every step (16 kyr) is effectively
quasi-static: what limits rigidity at plate scale is solver convergence, not E (see
design.md, phase 2).

**Plates** (`src/plates/`). A plate is a connected component of edges with
integrity > 0.5 between surface particles, found exactly on the CPU every 0.25 Myr from a
readback. Plate ids persist from one labelling to the next (each component inherits the
id most of its particles had), so splits and merges show up as events. Each plate gets
an Euler pole fitted to its particles' displacement over the interval, plus its rms
speed and the rms of the non-rigid part.

**Area, volume and melt** (`shaders/compute/area.wgsl`, `split.wgsl`; references
`src/columns/area.js`, `src/columns/volume.js`). A column's actual area is its cell in
a weighted Voronoi tessellation of the column positions, recomputed every step; the
cells tile the sphere exactly (boundaries through the contact points of touching
columns). Each column carries a crust **volume**; its thickness is volume / area, so
stretching thins crust and squeezing thickens it, with no rule beyond geometry.
Stretching also thins the lithosphere (its thermal age scales by (A₀/A)²), and where
the lithosphere is thinner than 70 km, new area brings decompression melt: 7 km of
crust per unit new area at a ridge, none under thick lithosphere. Melt is mixed into
the column's composition, so a stretched continent eventually becomes oceanic
(breakup). A column whose cell exceeds twice the base hexagon **splits** along the
cell's long axis into two halves that share its volume and reference area, joined by
an intact, stress-free edge. Sources (melt) and sinks (slabs) are logged, so the crust
volume budget closes. This replaced the old spreading stand-in (area growth over
diverging mantle).

**Subduction, suturing and arcs** (`law.wgsl`, `thermal.wgsl`, `flexure.wgsl`;
reference `src/edges/transitions.js`). Each edge has a timer that runs while it is
compressed beyond 50 MPa. If either column is denser than the mantle (negative
freeboard: ocean older than ~8 Myr), the denser one becomes a **slab** once the timer
passes 1 Myr (a tenth of that if it already touches a slab, so trenches propagate and
keep their polarity). If both are buoyant, nothing subducts; after 5 quiet Myr the edge
**welds** (suturing). A slab stays where it was and sinks at a Stokes rate set by its
negative buoyancy (~7 cm/yr for old ocean), warming and slowing; it is removed at
660 km. Its edges are measured in 3D, so as it sinks it stretches them and pulls the
plate behind it; its interface with the other plate is weak in compression. Crossing
80–160 km depth, a slab feeds an **arc**: surface columns within ~320 km gain 20% of
its crust volume (at continental density, 2750) and their lithosphere is reset hot.

### Initial conditions (`src/init/initial-state.js`)

1. N points scattered uniformly on the sphere, where N is chosen so the hexagons
   would tile 85% of it.
2. 100 sweeps of pure pairwise repulsion (Gauss–Seidel) push discs apart. The
   largest remaining overlap is shown in the panel; typical overlaps are ≪1%.
3. Continental vs oceanic crust comes from thresholding fractal (fBm) Perlin
   noise at the quantile that makes 35% of particles continental. Continental
   thickness rises from 30 km at the coast to 45 km in the interior.
4. Each particle bonds to its nearest neighbours within 1.3 contact distances
   (closest pairs first, at most 6 bonds each). A bond's rest length is always
   the sum of its two apothems, so positions and apothems are then fitted together
   (Gauss–Seidel) until every bond is at that length and no unbonded pair overlaps.
   The lattice starts stress-free (strain < 10⁻⁵); apothems end up within about
   0.65–1.45× nominal, and coverage rises to about 90%.
5. Bond damage comes from domain-warped Voronoi (Worley) noise around 12 random
   plate centres. Bonds near cell borders are damaged, which seeds a network of
   weak plate boundaries with triple junctions.
6. Velocities start at zero; the mantle is spun up separately (above).

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
│   │   └── layout.js          GPU struct definitions (Particle, Edge, ColumnProfile, GridNode) + slot-list layout
│   ├── geometry/
│   │   ├── icosphere.js       mantle grid mesh + adjacency
│   │   └── vec.js             vector / matrix helpers
│   ├── init/
│   │   ├── initial-state.js   the initial-condition recipe above
│   │   ├── noise.js           seeded Perlin noise, fBm, quantiles
│   │   ├── cell-list.js       CPU neighbour search
│   │   └── random.js          seeded RNG, uniform points on the sphere
│   ├── edges/
│   │   ├── topology.js        CPU reference for edge existence, slot order, storage checks
│   │   ├── law.js             CPU reference for the edge law: slices, Maxwell stresses, caps, integrity
│   │   └── transitions.js     CPU reference for subduction initiation, suturing, slab sinking
│   ├── columns/
│   │   ├── profile.js         CPU reference for column profiles: geotherm, Airy, GPE, strength
│   │   ├── flow.js            CPU reference for volumetric flow and the GPE push
│   │   ├── area.js            CPU reference for the tessellated area (weighted Voronoi cells)
│   │   ├── volume.js          CPU reference for crust volume, thinning, melt, breakup and splits
│   │   └── flexure.js         CPU reference for bending of the free radius (thin plate on a Winkler foundation)
│   ├── sim/
│   │   ├── params.js          every simulation parameter, with units
│   │   └── simulation.js      GPU buffers, compute pipelines, per-frame schedule
│   ├── mantle/
│   │   ├── mantle.js          lava-lamp mantle: buoyancy, screened poloidal flow, plumes, spin-up
│   │   └── sphere-mesh.js     cotangent Laplacian, CG Poisson solver, gradients, interpolation
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
│       ├── compute/           bindings.wgsl, locate, thermal, solver, edges, profile, law, flow, area, split, flexure
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
to `entryPoints` and to the schedule in `step` (`simulation.js`).

### Tests

```bash
npm run test:unit    # CPU code (zip, sampling, neighbour search, initial state, edge rules); seconds
npm run test:gpu     # mechanism tests against dist/plate-player.html; ~2 minutes
PLATE_PLANET=1 npm run test:gpu   # also the planet-scale tests; much longer
npm test             # both, rebuilding first
```

The GPU tests drive the built page in Chromium through `window.plate` (the same
object you can poke at in the browser console). They need Playwright with Chromium
(`npx playwright install chromium`). By default the browser uses SwiftShader, a CPU
implementation of WebGPU, so results don't depend on your graphics card; set
`PLATE_GPU=hardware` to use the real GPU. Headless Chromium can't present a WebGPU
canvas, so the browser runs with a window; on a machine with no display (a server,
CI), run `xvfb-run -a npm run test:gpu`.

**Keep GPU tests small.** Under SwiftShader every dispatch costs in proportion to
buffer *capacity*, not to how many particles are alive, and every readback takes
~0.5 s. A planet page (25 000 particle slots) steps in ~0.4 s and takes a minute to
open; an empty page with 1024 slots steps in ~0.2 s and opens in a few seconds. So
mechanism tests load a hand-built state into a small empty page, write inputs
straight into GPU buffers, and read back only when they must check something.
Tests on a generated planet are skipped unless `PLATE_PLANET=1`.

Page options (URL query): `?paused` starts paused, which scripts use to control
stepping exactly; `?empty` skips generating a world (load one with
`plate.loadState`); `?capacity=N` sizes the buffers for N particles (8 edge slots
each) instead of the planet default; `?grid=N` sets the hash grid's cells per axis
(coarse test worlds need larger cells). `plate.debug` holds the CPU reference
functions and readbacks checked against them: `plate.debug.edges()` (edge storage
and its invariants), `plate.debug.profiles()` (column profiles vs the reference,
worst difference per field) and `plate.debug.counters()` (crust volume sources and
sinks, breakups, clipped cells and capacity overflows since load).

**What belongs in the test suite.** Keep tests cheap and about *mechanisms*:
invariants (every slot free or active exactly once), small analytic cases (two
particles relaxing on a known time scale), file formats. Don't accumulate regression
tests on high-level results (plate speeds, land fraction, how a 100 Myr run looks).
They need lots of compute and pin down behaviour that is supposed to keep changing.
Measure those once when a phase lands, write the number in `docs/design.md`, and
move on.

## Debugging: treat causes as hypotheses

This project stacks many layers of numerical processing: noise fields, a packing
relaxation, a mantle solve, drag laws, a constraint solver with acceleration,
damage, thermal relaxation, sampling onto meshes. The distribution of intermediate
values at runtime is hard to predict, and intuition about it can be arbitrarily
wrong, whether from a bug, a subtlety of the math, or the particulars of a state.
Several findings in `docs/design.md` overturned the obvious explanation (for
example, more solver iterations made plates *weaker*, because a better-converged
solve enforced a conflict that a sloppy one ignored).

So when something looks wrong:

1. **State the suspected root cause as a hypothesis**, together with what it
   predicts you would see in the data, including what would refute it.
2. **Harvest runtime data to test it, with as little computation as possible.**
   Read the relevant buffers and look at distributions (quantiles, histograms,
   the worst few cases with their indices), not just means or the picture.
   `plate.readBuffers([...])` in the browser console returns raw GPU buffers;
   `unpack` in `src/gpu/layout.js` turns them into named arrays.
3. **Skip the warm-up.** Capture a snapshot of the state where the problem
   shows (`plate.snapshot()`, **Snapshot now**, or a record ZIP) and restart from it
   (`plate.restore(...)`, **Resume from record**) for every experiment, instead of
   re-running the minutes that lead up to it.
4. **Test pipeline stages in isolation.** Feed one stage harvested data and look
   at what it does: a CPU stage can be called directly from Node with arrays saved
   from a run; a GPU stage can be driven by `plate.loadState` with a small
   hand-built state (as the two-particle and chain tests do) or by a snapshot with
   one parameter changed.
5. **Write down what was measured,** including refuted hypotheses, in the findings
   sections of `docs/design.md`.

## Recording

Every run is recorded unless **Record this run** is unchecked (design.md §8.3 has
the full rationale):

- **Field series** every *Fields every* Myr (default 1): particle quantities averaged
  onto a fixed icosphere mesh (10 242 nodes, ~220 km spacing) with a Gaussian kernel
  one particle-spacing wide. Fields: surface height, crust thickness, density, age,
  temperature, east/north velocity (mm/yr), edge integrity, slab fraction, and
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
  free slot a splitting particle receives; which slot each new edge gets and its
  position in the slot lists, which change the order of floating-point sums). We note them; we don't engineer them away.
- The simulation advances one step per displayed frame, so simulated time per
  wall-clock second depends on the monitor's refresh rate and GPU speed. The
  panel reports steps and simulated Myr.

## To do

Ordered roughly by how much each item affects the plausibility of generated worlds.

**Physics**
- [x] ~~**Slab pull never fires.**~~ Phase F: buoyancy is the whole column's
      (freeboard), and pull goes through the slab's 3D-measured edges.
- [x] ~~**Slabs melt within a few frames.**~~ Phase F: slabs sink by their own
      buoyancy and warm over ~20 Myr; removed at 660 km.
- [x] ~~No ocean depth–age relation.~~ Phase B: Airy isostasy of the whole column
      profile, so sea floor deepens with √age, 2.8 km from ridge to old basin.
      Sea level moved from 3.5 to 2.6 km in the elevation datum to keep
      35 km continents ~0.8 km above it.
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
- [x] ~~Long steps tripped the browser's GPU watchdog.~~ Steps are submitted in
      pieces of 16 solver iterations, waiting for each.
- [x] ~~Growth recreated bond/contact conflicts~~ (a growing particle's bonds
      kept their rest length while collisions pushed to the new apothem sum). A
      bond's rest length is now *always* the sum of its particles' apothems,
      derived rather than stored, so bonds and contacts agree on the touching
      distance whatever changes a particle's area. The initial state fits positions
      and apothems together so every bond starts at that length (strain < 10⁻⁵).
      To check: whether growth and splitting still tear plates (design.md,
      driven-plate measurements).
- [x] ~~A compressed bonded pair was also a contact.~~ Phase 3: bonds and contacts
      are one edge array, and contacts skip bonded pairs.
- [x] ~~Fully damaged bonds never died.~~ They are retired (phase 3) and the pair
      interacts by contact.
- [x] ~~Contacts were rebuilt every step.~~ Phase A: one kind of edge, keyed by pair,
      persistent while in range (design 3.2). Stress-based subduction initiation,
      suturing and friction can now keep state on the edge.
- [x] ~~**Cold plates never weld.**~~ Phase F: buoyant pairs in sustained, quiet
      compression weld (suturing).
- [ ] **No ridge push yet.** The GPE push exists but is off until basal drag is
      calibrated against it (design, phase D findings).
- [x] ~~**Subduction rarely starts.**~~ Phase F: initiation is a timer on compressed
      edges with a dense side, not an overlap.
- [ ] **No trench or outer rise:** the slab pulls the plate in-plane only; its
      vertical load on the plate behind it is not coupled to the bending.
- [ ] **No friction.** Edges have strength caps in compression and tension
      (phase C) but no tangential part, so sliding columns feel no friction. A
      tangential channel needs a per-particle orientation (design §10).
- [x] ~~**Slabs detach from their plate.**~~ Phase F: a slab subducts where it is
      and keeps its edges until it has sunk below the plate behind it.
- [x] ~~Water loading factor~~ was d·(1 + ρ_w/ρ_m); now ρ_w/(ρ_m − ρ_w) (phase 3).
- [x] ~~**Mass is not conserved at splits.**~~ Phase E: volume is state, shared
      equally at splits; the child copies the parent's material.
- [x] ~~**Subduction choice.**~~ Phase F: only negatively buoyant columns subduct
      (the denser of a pair); buoyant pairs weld instead.
- [x] ~~The Swift–Hohenberg mantle took hundreds of Myr to emerge, locked into a
      fixed pattern once it did, and its velocity scale was arbitrary.~~ Replaced by
      the lava-lamp mantle: spun up alone, never settles, speed calibrated to ~4 cm/yr.
- [ ] **Crust → mantle feedback** is not coupled: continents should blanket (warm)
      the mantle beneath them, and slabs should be the main cold downwellings,
      replacing the random cold drips.
- [x] ~~**The spreading proxy stresses plates.**~~ Removed in phase E: new crust now
      comes from stretching (tessellated area at fixed volume) plus decompression melt
      where the lithosphere is thin.
- [ ] **No young ridges in the initial state:** its oceans are all 60 Myr old, so
      nothing melts until stretched by β > 1.4 (design.md, phase E findings).
- [ ] **Breakup needs β ≈ 10** with melt mixed by volume; real margins break at
      β ≈ 3–5. Decide the criterion at integration.
- [ ] **Residual non-rigidity** of plate interiors (~1 mm/yr against ~0.1 mm/yr for
      ideal elastic plates) is solver under-convergence: measure at higher resolution
      before changing the solver.
- [x] ~~Mantle flow is curl-free (no toroidal flow).~~ That is correct for the flow
      *beneath* the plates (buoyancy drives poloidal flow); Earth's toroidal surface
      motion comes from the plates themselves, through rigidity and boundaries.

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
      the `> −50` guard, so only near-mantle-temperature edges heal.

**Performance**
- [x] ~~`nearest_node` brute force over all mantle nodes.~~ Each particle caches
      its node and updates it by a greedy walk over the icosphere (`locate.wgsl`).
- [ ] Every dispatch covers the full capacity (25 000 particles, 131 072 edges)
      regardless of how many are active. An indirect dispatch on the live count
      would help.
- [ ] Neighbour-search hash cells (3/32 ≈ 0.094) are about 2.7× the edge range;
      cells ≈ edge range would scan about 10× fewer candidates.
- [ ] Decouple simulation steps from display frames (fixed steps per frame, or a
      step budget) so speed doesn't depend on the monitor.

**For game use**
- [ ] Export: rasterise a record's `surface_height` to an equirectangular height
      map and save as PNG (the record already has the data on a mesh).
- [ ] Expose `INIT_DEFAULTS` (continent fraction, plate count, etc.) in the UI.
