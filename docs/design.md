# Plate Player — design

This document describes the model as it is, and why it is that way (sections 1–8).
Specifications of work not yet built are collected in section 9; questions to settle
are in section 10.

**Status.** All the physics of one pair-interaction law is assembled:
- persistent edges with graded, history-dependent integrity;
- column profiles;
- an edge law with depth slices, a normal and a tangential (friction) channel, and
  face rolling;
- solved column rotations and reference shapes;
- work-driven damage with thermal healing;
- exact area and crust-volume budgets with melt and splits;
- subduction and suturing as edge-state transitions, sinking slabs, arcs, and
  bending of the free radius.

Each piece is checked on its own against a known answer. The next step is
**integration**: the first planet runs of the assembled model and joint calibration
(section 9.2).

**How to use this document.** It is a working description, not a specification.
Don't pursue its goals too doggedly: when a step runs into difficult problems (a
method that doesn't converge, a cost that doesn't fit, a requirement that turns out
to fight another), stop and discuss a redesign rather than forcing the plan through.

Contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Principles](#2-principles)
3. [State](#3-state-what-the-gpu-holds)
4. [Physics: one energy, one dissipation](#4-physics-one-energy-one-dissipation)
5. [The time step](#5-the-time-step)
6. [Boundaries, subduction and spreading](#6-boundaries-subduction-and-spreading)
7. [Resolution](#7-resolution)
8. [Records and contracts between scales](#8-records-and-contracts-between-scales)
9. [Development plan and extensions](#9-development-plan-and-extensions)
10. [Open questions](#10-open-questions)
11. [Glossary](#11-glossary)

---

## 1. Goals and non-goals

**Goals**

- Generate plausible planets for a game: continents, ocean basins, mountain belts,
  trenches, rifts, and their evolution over hundreds of Myr.
- Physics that is *coherent*: every force comes from one energy or one dissipation,
  so the behaviours agree with each other and with the method's stability guarantee.
- Designed for **~50 km particle spacing (~240k particles)**, with graceful
  degradation at coarser resolution (~220 km, ~10k particles; today's default) for
  fast iteration. "Graceful" means it still runs and still looks tectonic. Statistics
  like trench widths may deviate from Earth's.
- One member of a **stack of models** at decreasing scale. Each runs in isolation
  and communicates only through recorded data (section 8).
- Reproducible *enough*: a seed regenerates the initial world, a manifest records the
  code version and parameters, and snapshots resume runs. Bit-identical GPU results
  are explicitly **not** a goal; on GPUs they cost more than they are worth.

**Non-goals (for this level)**

- Inertia. Plate tectonics is overdamped (Reynolds number ~10⁻²⁰).
- Two-way coupling between scales. Information flows down only.
- Surface processes (erosion, sedimentation, climate) and earthquakes. These belong
  to lower levels or separate models; this level exports what they need.
- Geological history and provenance. They are deferred, but the design leaves room:
  per-particle fields and an event log (section 8).

---

## 2. Principles

1. **Overdamped implicit steps.** Velocity is set by force balance, not by
   integrating accelerations. Each step minimises

   > Φ(x) = Σᵢ (γᵢ / 2dt) |xᵢ − x̃ᵢ|² + U(x)

   where γᵢ is basal drag, x̃ᵢ is where the mantle alone would carry particle i,
   and U is the total potential energy. This is implicit Euler for overdamped
   dynamics, and the Onsager variational principle in discrete form.
   **Extended position-based dynamics (XPBD) solves exactly this problem with
   drag in place of mass** (section 5.2).

2. **Everything else is integrated unconditionally stably.** Relaxations
   (temperature, damage healing) use exact exponentials, x ← x∞ + (x − x∞)e^(−k dt).
   Irreversible changes (yield, damage, topology) are applied as corrections between
   solves. No explicit rate may have a stability limit on dt.

3. **One pair-interaction law, no identities.** Every interaction between two
   columns is a function of the two columns' state and the state of the edge between
   them, and nothing else. There are no bond/contact kinds, and no plate or continent
   id enters any force. Bonds, contacts, faults, ridges and subduction interfaces are
   regions of one continuum (section 4.5). Plate ids and boundary classes are
   *diagnostics* derived from edge state and kinematics, for display, records and
   events.

4. **Exact budgets.** Surface area is exact by construction: each column's actual
   area is its cell in the tessellation of the markers, so the cells tile the sphere.
   Crust volume changes only at named, logged sources (melt, arcs) and sinks (slabs).
   Deformation moves area and volume between columns; it never creates or loses them.

5. **Material parameters are continuum quantities.** Young's modulus, Poisson's
   ratio, viscosities and densities are given in physical units and converted to
   per-particle and per-edge constants using the local particle size. The material
   must not change when resolution changes.

6. **Levels talk only through records.** A model never reads another model's
   internals; it reads a *forcing* built from a parent's *record* or from a
   procedural generator, under a stated contract (section 8).

7. **Seeded randomness only.**

8. **Every component has a CPU reference and is inspectable on its own** (section
   9.1). Each physics kernel has a pure JavaScript reference next to it; tests check
   the reference against an answer known without the rest of the model, and the
   kernel against the reference.

---

## 3. State: what the GPU holds

Sizes at the 50 km target: ~240k particles, ~1M edges. Memory stays within a few
hundred MB of storage buffers. Struct layouts are declared once in
`src/gpu/layout.js` (the WGSL structs and the JavaScript offsets are both generated
from it); parameters, with units, in `src/sim/params.js`.

### 3.1 Particle = lithospheric column

A particle is a prism of lithosphere. Its *position* is the direction of the centre
of its top surface; its *elevation* (the free radius) is a separate coordinate.

| Field | Units | Notes |
|---|---|---|
| `position` | unit sphere | direction of the column |
| `elevation` | km | top of the column above the mantle datum: the **free radius**, a separate coordinate rather than \|x\| (4.4) |
| `state` | – | 1 surface, 2 slab (sinking below the surface, section 6) |
| `body_axis`, `spin`, `rotation_total` | unit tangent, rad, rad | the column's body x-axis, carried with it and turned by its spin; the spin is its rotation about the vertical this step, a solved unknown (4.2); `rotation_total` accumulates it (paleomagnetic declination) |
| `area_ref`, `apothem` | unit sphere², unit sphere | stress-free (reference) area; apothem a = √(area_ref/2√3) is the mean radius. Changed only by inelastic dilatation (4.7) and splits (4.8) |
| `shape_p`, `shape_q` | – | reference shape: traceless log-stretch L = [[p, q], [q, −p]] in the body frame; aspect ratio e^(2\|(p,q)\|) (4.2) |
| `cell_area`, `cell_area_prev`, `cell_shape`, `cell_clipped` | | derived each step: the column's cell in the tessellation (4.8), last step's area, the cell's centroid and major axis, and whether it touched its clip polygon |
| `crust_volume` | km·unit sphere² | crust volume V. Changes only at sources and sinks (4.8) |
| `thickness_km` | km | derived: V / cell_area |
| `composition_density` | kg/m³ | crust composition, at mantle temperature; < 2850 is continental |
| `thermal_age_myr` | Myr | sets the lithosphere, 2.32√(κt) capped by composition, oceans and continents alike; scaled by stretching, reset by arcs (4.8, 6) |
| `age` | Myr | time since the crust formed (reset at breakup) |
| `temperature` | K | surface temperature, for display only; the thermal structure comes from `thermal_age` |
| `slab_trailing`, `slab_neighbours`, `arc_count`, `arc_supply` | | subduction bookkeeping (section 6) |
| `flexural_rigidity`, `bending_moment`, `elevation_start` | | bending solve (4.3) |
| `predicted_pos`, `iter_start`, `iter_prev`, `velocity` | | solver scratch; velocity is a diagnostic |

Per particle there is also a scratch record (`ParticleDelta`, 48 words): position
and spin corrections accumulated with integer atomics, the slot list of its edges
(≤12, sorted by partner angle), and its tessellation neighbours with bending
weights. It is an index rebuilt every step, not state.

### 3.2 Edge = the interface between two neighbouring columns

There is one kind of edge. Whether a pair behaves like a weld, a fault, a gravel
contact or a ridge is a matter of the edge's state and the two columns' profiles
(section 4.5), not of a type field.

| Field | Notes |
|---|---|
| `particle_a`, `particle_b` | the pair is the edge's identity; it persists across steps |
| `integrity` φ | 0..1. Scales cohesion. Lowered by plastic work, raised by thermally activated healing (4.7) |
| `length`, `direction_x/y/z` | the pair's distance and separation vector at the last update. Strain and sliding are measured from them, so a change of rest length or shape adds no stress |
| `slice_stress_mpa`, `slice_shear_mpa` | the normal and tangential Maxwell stress of each of 8 depth slices of the shared interval: the memory that makes the edge history-dependent (4.5) |
| `moment` | the face moment against relative rotation of the two columns (4.5) |
| `timer` | sustained-compression accumulator for suturing (section 6) |
| `born_myr` | creation time |
| `rest`, `lever_a`, `face_length`, stiffnesses, reference offsets, multiplier bounds | derived each step by `edges_law` for the solve (4.5) |
| `lambda`, `lambda_prev`, `lambda_t`, `lambda_r` | XPBD multipliers (normal, tangential, rolling); Chebyshev needs the previous normal one |
| `strain`, `inelastic_strain`, `inelastic_shear`, `slip`, `work_mpa_km`, `stress_normal_mpa`, `stress_shear_mpa`, `overlap_km` | diagnostics of the last update |

**Existence is purely geometric.** An edge exists while the two columns are near in
map view (gap below `edge_range_margin` 0.1 of their rest length, the two radii
along the pair) *and* their depth intervals overlap. It is created, at φ = 0
(except the parent–child edge of a split, at 1), when the pair comes into range,
and removed when it leaves range. Neighbour exchange (a pair leaving range while
another enters) is how the network flows past the reference shapes' limit, as T1
events do in a foam.

**Storage.** Edges live in one free-listed array (131 072 slots) and persist across
steps. Per step, on predicted positions: pairs out of range (or with a dead
particle) are pruned, and the survivors are listed at both ends; each particle i
creates edges to in-range partners j > i that it does not already list; each slot
list is sorted by angle. No pair is ever created twice: pair (i, j) is created only
by thread i, after checking i's survivors, and edges that other threads add to i's
list meanwhile have partners below i. Prune (pushes to the free list) and create
(pops) are separate dispatches. Snapshots hold only the edge array.

### 3.3 Plates (diagnostic only)

Plates are connected components of edges with integrity above 0.5, labelled on the
CPU by union-find every 0.25 Myr (milliseconds for 10⁴ particles; `src/plates/`). Ids
persist by majority inheritance, which yields `plate_split`, `plate_merge` and
`plate_lost` events. Each plate gets an Euler-pole rotation fitted to its members'
displacements over the labelling interval (a finite rotation, not the one-step
velocity, which at ~1 cm/yr is comparable to the solver's tolerance). Plates feed
records, events and display, never forces.

### 3.4 Mantle

The mantle is a **basal-forcing generator** (section 8) on the CPU, the "lava lamp"
model in `src/mantle/`. Its requirements: to be alive (evolving, never locking into
one flow map), to have realistic flow speeds beneath the plates, and to start from a
settled state, without solving mantle fluid dynamics.

- **State:** a buoyancy field b (mid-mantle temperature anomaly) on a 2562-node
  icosphere (~480 km spacing).
- **Flow:** poloidal, v = s∇Φ with (∇² − 1/ℓ²)Φ = b: hot regions diverge (upwell),
  cold regions converge. Flow beneath plates being poloidal is physical: buoyancy
  drives poloidal flow, and Earth's toroidal surface motion comes from the plates'
  rigidity and boundaries (established). The screening length ℓ is the mantle depth
  (2900 km), because flow driven by an anomaly in a layer of depth H dies off over ~H
  (modelling choice). A consequence: the flow has no vertical vorticity, so basal drag
  resists a column's spin but never drives it.
- **Evolution:** semi-Lagrangian advection of b by its own flow (upwellings broaden,
  downwellings narrow, as in convection), a small effective diffusion, decay with a
  plume lifetime of 120 Myr, and Poisson births of hot plumes and cold drips
  (0.3/Myr each, so ~35 active, like Earth's ~40 hotspots; order of magnitude).
- **Speed calibration:** the scale s comes from the plume statistics, not tuning. The
  expected mean-square flow is Σ rate·(lifetime/2)·E/(4π), where E is one plume's
  flow energy (one screened solve per plume kind at start-up). Advection sharpens
  downwellings, so the realised speed is higher by a measured factor of 1.69 (6 seeds
  × 400 Myr), which is divided out. Target long-run rms 4 cm/yr; realised roughly
  ±20% around it per seed; the instantaneous rms fluctuates ~2× over 100 Myr.
- **Spin-up:** 400 Myr (over three plume lifetimes) alone before the crust starts,
  in ~2 s.
- **Alive:** the buoyancy pattern decorrelates in ~100 Myr (correlation 0.35 after
  50 Myr, ~0 after 100). Unit-tested as correlation < 0.5 after 250 Myr.
- The mantle steps on the CPU every `stepMyr` and its flow is uploaded to the GPU
  grid; each particle tracks its nearest node by a downhill walk over the icosphere
  (exact on a Delaunay mesh; usually zero or one hop per step).
- **Not coupled:** crust → mantle feedback (section 9.3).

### 3.5 Initial state

`src/init/initial-state.js`, every random choice from one seed:
1. N points scattered uniformly and relaxed by pairwise repulsion, with N chosen so
   the hexagons would cover 85% of the sphere.
2. Crust type from thresholded fractal noise: 35% continental (2700 kg/m³, 30–45 km
   from coast to interior, age 1000 Myr); the rest oceanic (3000 kg/m³, 7 km,
   60 Myr).
3. Bonds to nearest neighbours (≤6). Positions and apothems are fitted together so
   every bond starts at its rest length (strain < 10⁻⁵) and no unbonded pair
   overlaps.
4. Plates: a domain-warped Voronoi pattern of 12 plate centres. Bonds near cell
   borders are damaged, and bonds joining different cells start fully damaged, which
   seeds a network of weak boundaries with triple junctions.

A loaded state is declared stress-free: edges are loaded from their rest length,
and a patch built flat on the sphere would otherwise start pre-stretched.

---

## 4. Physics: one energy, one dissipation

Notation: A = column area, hₑ = elastic thickness, E = Young's modulus (70 GPa),
ν = Poisson's ratio, g = gravity, ρ_m = mantle density (3300 kg/m³).

### 4.0 Column profile (what each column contributes to its edges)

Each column carries a vertical profile, recomputed every step from its own state
(`column_profile` in `profile.wgsl`; reference `src/columns/profile.js`). It is a
derived buffer, never saved. 8 slices: 3 over the crust, 5 over the mantle
lithosphere, spanning [0, max(L, crust)]. Per slice: depth, thickness, T, ρ, P,
log₁₀η, ductile strength. Per column: lithosphere base L, Airy elevation and the
unloaded freeboard, GPE, surface heat flow, Moho temperature, and the shallowest
brittle–ductile transition at φ = 1. Nothing downstream reads crust type directly,
so a stretched continent that turns oceanic changes behaviour through its profile
alone.

| Piece | Choice | Why |
|---|---|---|
| Oceanic geotherm | half-space cooling, T = Tₛ + ΔT erf(z / 2√(κt)); L = 2.32√(κt) (the 90% isotherm); thermal age capped where L = 100 km (~59 Myr) | the depth–age law directly, and a cheap stand-in for the plate model's flattening |
| Continental geotherm | steady conduction, heat production 1.5 µW/m³ in the top 15 km, L from thermal age capped at 150 km, gradient continued below L to mantle temperature | 41 mW/m² surface heat flow (cratonic) and a 334 °C Moho under 35 km of crust; thermal age so stretching can thin it |
| Density | composition density at mantle temperature × (1 + α(Tₘ − T)) | thermal contraction of crust and mantle alike |
| Airy | freeboard = ∫(1 − ρ/ρₘ)dz down to where T = Tₘ; water-loaded below sea level | whole column, so the lithospheric mantle's contraction sets depth vs age |
| GPE | Γ = g ∫(ρ − ρ_ref)(r − r_b)dr, ρ_ref = mantle below the datum and 0 above, at the Airy height | independent of the base r_b in equilibrium, so no compensation-depth parameter |
| Brittle | τ = φc + μ(1 − λ_pore)P, μ = 0.6, λ_pore = 0.4 (hydrostatic), c = 20 MPa | Byerlee friction. φ belongs to the edge, so the profile stores P and the edge law applies it |
| Ductile | dislocation creep at ε̇ = 10⁻¹⁵ s⁻¹: wet quartzite (Gleason & Tullis 1995), diabase (Mackwell et al. 1998), dry olivine (Hirth & Kohlstedt 2003); shear strength = σ_diff/2, η = strength/2ε̇ | laboratory laws. Constants are from memory of the literature and should be checked against the papers before calibration; order-one factors between uniaxial laws and tensor invariants are ignored |

Depth integrals (Airy, GPE, P) do not use the 8 slices: they use 2-point
Gauss–Legendre quadrature on 6 crust and 16 mantle sub-intervals, down to the depth
where T reaches mantle temperature, so the slices only resolve what the edge law
needs.

- **Hypsometry:** ridge 2.85 km deep, old ocean 5.62 km, 35 km continent +0.81 km
  (Earth: ~2.5, ~5.7, ~0.8), with sea level at 2.6 km in the elevation datum; 30 km
  of continental crust is a shelf just under water. Sea level is the one calibrated
  number here.
- **Continents are "jelly sandwiches":** the strongest point is brittle upper mantle
  at ~70 km, below a weak quartz lower crust. The edge law integrates the whole
  envelope, so continental stiffness depends on the Moho temperature.
- **At thermal age 0** the column is at mantle temperature throughout (no
  lithosphere, no brittle layer); the reported heat flow uses a 1 kyr floor on age.
- **Checked:** the geotherms satisfy their heat equations; subsidence below the
  ridge matches 2ρₘαΔT√(κt/π)/(ρₘ − ρ_w) (Turcotte & Schubert) to 2×10⁻⁶ for ages
  1–150 Myr; GPE differences equal the half-space ridge push to 3×10⁻⁷
  (4.2×10¹² N/m at 100 Myr); moving the integration base 100 km deeper changes GPE
  by ≤ 1.2 MPa·km (ridge push is ~2000); the envelope peaks where the curves cross
  (olivine ~650 °C, quartzite ~260 °C); the kernel matches the reference to
  ≤ 2×10⁻⁵ of each field's scale.

### 4.1 Dissipation (the "mass" of the solver)

**Basal drag: Couette shear across the asthenosphere.** Each column is dragged by
traction η_a(v_mantle − v_plate)/h, h the thickness of the low-viscosity channel
between the plate's base (the lithosphere–asthenosphere boundary, LAB) and the depth
where viscosity rises again. In the solver this is γ = A·η_a/h (`inverse_drag`), so
plate speed is *not* slaved to the mantle: it comes out of the force balance between
drag, the edges, slab pull and (when switched on) ridge push.

| Ingredient | Value | Status |
|---|---|---|
| Asthenosphere viscosity η_a | 10¹⁹ Pa·s (range 10¹⁸–10²⁰) | established range; **the effective drag coefficient to tune** |
| Base of the low-viscosity layer | 300 km (channel ≥ 50 km) | established order (~200–400 km); modelling choice |
| LAB depth | from thermal age, capped at 100 km (oceanic) and 150 km (continental; cratons reach ~250) | established order |
| Couette (linear shear) profile | | modelling choice; its consequence, that deep keels couple more strongly, is a common argument in the literature |
| Plate speeds today | 1–10 cm/yr, rms ~4 cm/yr; fast plates are those with slabs (Forsyth & Uyeda 1975) | established; the calibration target |

**Spin drag.** The same basal shear resists a column's spin: drag × its squared
radius of gyration, area·cosh(2s)/2π for the equal-area ellipse (`inverse_spin_drag`).
Summed over a plate, spin and orbital drag are the two terms of the parallel-axis
theorem: a plate turning rigidly feels both.

**Slabs** are dragged in the plane like surface columns, and sink at a Stokes-like
rate from their buoyancy (section 6).

### 4.2 In-plane elasticity, rotation and shape

The edges carry the in-plane mechanics (4.5); there is no separate per-column
elastic energy, so elasticity is not counted twice. A column's deformation energy is
half of each of its faces'.

**Lattice elasticity.** On a triangular lattice whose columns may rotate, normal and
tangential edge stiffnesses k_n and k_t give isotropic plane elasticity with energy
per area √3[k_n tr(ε)²/4 + (k_n + k_t) ε_dev:ε_dev/4]:

> K₂ = √3 k_n / 2,  μ = √3 (k_n + k_t) / 4,  ν = (k_n − k_t)/(3k_n + k_t),  μ = E/2(1 + ν)

`shear_stiffness_ratio` k_t/k_n = 0.2 gives ν = 0.25 (normal springs alone force
1/3). The normal stiffness is set so the lattice's Young's modulus stays E·h for any
ratio: k_n = E h (3 + r)/(2√3(1 + r)) (`bondRate`, `normalStiffnessPerModulus`).
Pair interactions cannot give ν above 1/3; a thin viscous sheet behaves like ν = ½,
and the missing compressibility is supplied by the area outlet (4.7), which is where
thickening happens.

**Rotation is solved for.** Each column's spin (rotation about the vertical per step)
is an unknown of the XPBD solve next to its position, resisted by spin drag (4.1).
The edges' tangential channel couples spins and positions. Rigid rotation of a patch
turns every column with it and stores no shear; a tangential spring between centres
without column rotations would resist rigid rotation. The body axis is carried to
each new tangent plane by projection (parallel transport to first order) and turned
by the spin.

**The neighbour fit measures inelastic strain** (`columns_flow` in `flow.wgsl`;
reference `fitStrain` in `src/columns/shape.js`). Each step a column's distributed
inelastic strain increment is fitted by least squares from its surface edges, two
equations each, in its body frame:

> n·ε·n = inelastic normal strain,  t·ε·n + w = ductile tangential strain

The fourth unknown w is the neighbourhood's rotation relative to the column's spin:
the micro-rotation (spin) and macro-rotation of a Cosserat medium may differ, and a
column whose spin lags must not read the lag as shear. The fit's trace changes
area_ref (4.7); its deviatoric part changes the shape. Increments are per length,
not per rest length (they differ where a shape has met its cap).

**Reference shape.** The hexagon of area area_ref stretched by U = exp(L), L
traceless in the body frame: principal radii a·e^(±s), s = |(p, q)|, long axis at
½atan2(q, p) from the body axis. The radius along a unit direction n is
a/√(nᵀU⁻²n), and the rest length of a pair is the two radii along it, so an affinely
stretched lattice keeps every rest length exact. The shape is a finite-strain
ellipse, a material marker:
- it turns with the material: with the body frame (spin) plus w;
- it accumulates the deviatoric strain, up to `shape_aspect_cap` 3; beyond the cap,
  deviatoric flow is taken up by neighbour exchange;
- faults (brittle slip on an edge) do not change it: they are jumps between columns;
- it sets rest lengths, range and contact, the levers of the tangential channel, the
  spin drag, and the split axis (4.8).

The tessellation keeps scalar weights (the apothem): anisotropic weights would curve
cell boundaries and break the exact tiling. The cell's measured shape against the
reference shape is a diagnostic.

**Checked:** for k_t/k_n = 0, 0.2, 1 and three lattice orientations, the lattice
energy (spins at their optimum, the mean sliding) equals (K₂/2)tr² + μ ε_dev:ε_dev to
10⁻⁹; rigid rotation costs nothing with free spins and 3(k_t/2)L²ω² per cell with
spins held. On the GPU a bonded 19-column patch turned by 0.02 rad in one step
keeps 0.01892 rad of the turn, against the parallel-axis prediction 0.01897 (orbital
drag Σr² against spin drag Σ area/2π), every column spinning with it within 0.35%.
Radii are exact (10⁻¹²) for an affinely stretched hexagon. Hot columns stretched by
hand to 1.838× have aspect 1.866 and area 1.848× the base, with no lattice pair
leaving range; in simple shear to 1 the aspect is 2.626–2.636 against the
finite-strain ellipse's 2.618, and it stops at the cap with area within 2%.

### 4.3 Bending

The elevation field is a thin elastic plate on a fluid foundation, discretised on the
tessellation (`columns_flexure_begin`, `columns_moment`, `columns_flexure` in
`flexure.wgsl`; reference `src/columns/flexure.js`). Per step the elevations w
minimise

> Σᵢ Aᵢ[(Kᵢτᵢ/2dt)(wᵢ − w₀,ᵢ)² + (Kᵢ/2)(wᵢ − w_eq,ᵢ)²] + Σᵢ (DᵢAᵢ/2) mᵢ²,  mᵢ = (1/Aᵢ) Σⱼ cᵢⱼ (wⱼ − wᵢ)

- cᵢⱼ = shared cell boundary / distance: the finite-volume ("cotangent") Laplacian of
  a Voronoi mesh.
- The first term is implicit Euler against viscous rebound (Haskell time τ from w₀,
  the step's starting elevation), the second the Winkler foundation (4.4), the third
  thin-plate bending (D/2)(∇²w)². The Gaussian-curvature part of the Kirchhoff energy
  only adds boundary terms and is left out.
- D = E hₑ³/12(1 − ν²), hₑ = Σ slices × their Maxwell factor this step: the same
  elastic fraction the edge law uses in-plane. `flexure_factor` scales D (0 = Airy).
- Solved by damped Jacobi sweeps (bending part of the diagonal × `flexure_damping`
  2), `flexure_iterations` 8 per step. With D = 0 one sweep is exactly per-column
  relaxation to Airy. Sweeps needed grow as (α/ℓ)⁴ (α the flexural length, ℓ the
  spacing): few at today's 220 km (α/ℓ < 1), ~2000 at α/ℓ = 3.4 (section 9.3).
- Slabs are not in the network (they sink, section 6). Bending the elevation field
  needs no extra unknowns; directors with a Kirchhoff penalty, the alternative, are
  not needed while this passes the flexure benchmark.

**Checked:** on a strip of 55 tessellated columns (α = 109 km = 3.4 spacings) under a
line load, peak 0.1506 km against 0.1485 analytic, first zero at 2.34α (3π/4 = 2.36),
forebulge 4.1% of the peak (e^(−π) = 4.3%), worst deviation 1.4% of the peak; the GPU
is within 0.03% of the exact (dense) solve.

### 4.4 Buoyancy and the free radius

Each surface column has an equilibrium height w_eq from Airy isostasy of its whole
column (crust and lithospheric mantle with temperature-dependent densities; 4.0),
with water loading Δρ = ρ_m − ρ_w where submerged. The restoring term
½ Δρ g A (w − w_eq)² is a Winkler foundation; with D = 0 every column sits at w_eq,
with D > 0 loads spread over the flexural length (foreland basins, forebulges). The
approach to equilibrium is limited by the viscous rebound time of a column-sized load
(Haskell), τ = 4πη/(Δρ g λ), λ = 4 apothems, η = 10²¹ Pa·s (upper mantle, from
post-glacial rebound): ~30 kyr at the base particle size, so columns track their
equilibrium closely.

**The radius is a separate coordinate,** not |x|. In-plane strain is measured on
directions on the unit sphere: a column 1 km higher would otherwise have a chord
1/6371 longer, which reads as 11 MPa per km of relief at 70 GPa. Edges with a slab
end are the exception: they are measured in 3D (section 6).

Slab columns have no restoring equilibrium; they sink (section 6).

### 4.5 Pair interaction: the edge law

Given an edge (i, j) and the two column profiles (4.0), integrate over the depth
interval the two columns share. Depth slices act **in parallel** (columns are
vertically rigid: the thin-sheet assumption); each slice is a **Maxwell element**
(spring in series with dashpot η(z)), integrated by implicit Euler. Kernels:
`edges_law` (before the solve) and `edges_maxwell` (after) in `law.wgsl`; reference
`src/edges/law.js`. *Prior art:* the bonded-particle model of rock mechanics in the
discrete element method (Potyondy & Cundall 2004), where bonds break into frictional
contacts; here the two are one edge with graded integrity.

**Slices.** 8 equal slices over the shared interval (from the higher column top down
to the shallower column base; a column spans max(lithosphere, crust)). At each slice,
each column's temperature and creep strength are evaluated at its own depth;
viscosities and ductile strengths combine by harmonic mean, pressures by arithmetic
mean. Memory is per slice, because one integrated number cannot decay correctly when
cold and hot slices mix. Per slice the Maxwell factor r = 1/(1 + E dt/η): cold slices
act as springs (r ≈ 1), hot ones as dashpots (r ≈ 0), so the **elastic thickness
emerges** (47 km of 100 km for a pair of 100 Myr ocean).

**Geometry.** d = |xᵢ − xⱼ| on the unit sphere; rest length = the two columns' radii
along the pair (their reference shapes, 4.2), the apothem sum for round columns;
contact point at the lever fraction rᵢ/rest from i. Normal strain increments are
(d − length)/rest, from the edge's stored length.

**Normal channel.** Stress σ' = (σ + E dε) r per slice, then capped:

| Direction | Brittle slices | Ductile slices |
|---|---|---|
| compression | μ(1 − λ_pore)P + φc | flow stress |
| tension | φ(c + μ(1 − λ_pore)P)/q | flow stress (sign-blind: hot rock resists separation), × (1 − `ductile_damage_weakening`(1 − φ)) |

A slice is brittle where friction is weaker than creep. Slice stresses are deviatoric
(measured from the lithostatic state), so a brittle slice in extension fails by
normal faulting, at Anderson's fraction 1/q of its frictional strength,
q = ((1 + μ²)^½ + μ)² (3.12 at μ = 0.6): about 6 MPa at the surface, growing with
depth. (Until the trench hinge, the cap was cohesion alone, φc: a crack in absolute
tension. A plate could then transmit at most ~2×10¹² N/m, too little to be pulled by
a slab against any trench contact; section 6.) Above a cap: ductile slices
flow and brittle slices slip in compression (the stored stress is the capped
stress). Brittle slices *crack* in tension: they store their elastic trial stress,
less only the creep their capped stress drives, so they stay referenced to the rest
length. Cracks close, and contacts engage, exactly at touching. So an edge at φ = 0
between cold columns only pushes (a contact). A new edge loads every slice
elastically from the rest length, σ = E·(strain now): a gap is an open crack, an
overlap pushes at once.

**Tangential channel.** Each slice also carries a tangential Maxwell stress, stiffness
G_t = `shear_stiffness_ratio`·E, the same Maxwell factor. Its strain is the faces'
sliding since the last update, per rest length:

> δ_t = L₀ψ − l_a θ_a − l_b θ_b

ψ the rotation of the pair line about the outward radial (from the stored
direction), θ the spins, l the levers to the contact point (l_a + l_b = L₀). Rigid
rotation (ψ = θ_a = θ_b) slides nothing; so does any rigid rotation of the sphere, to
the curvature over one spacing.
- **Caps per slice, symmetric:** friction φc + μ·max((1 − λ_pore)P − σ_n, 0), σ_n the
  slice's stored normal stress (+ tension): compression across the face strengthens
  it, tension weakens it, an open crack is frictionless. Or the ductile flow stress,
  whichever is weaker.
- **Above the cap:** a brittle slice slips (`slip`: a fault, a jump between the
  columns); a ductile slice flows (`inelastic_shear`: distributed, it changes the
  shapes, 4.2). The stiffest slice decides which, as for the normal channel.
- **Only between surface columns;** slab interfaces are normal-only.

**Face rolling.** Flat faces resist relative rotation of the two columns: a relative
spin Δθ opens one end of the shared face and closes the other, moment
M = K·face²/(12·rest)·Δθ (K = Σ E r dz the normal stiffness, face the shared
tessellation boundary; `rolling_stiffness_factor` 1). It decays with the mean Maxwell
factor and is capped by the kern rule of masonry: the face opens at one end once
|M| > (compression + tension cap)·face/6. Without it a column in a slipping zone could
spin like a ball bearing and erase friction. Its length scale is the spacing, so it
vanishes in the continuum limit.

**In the solve.** Per step each row's law is linear (the Chebyshev requirement):
S = S₀ + K(ε − ε₀), with S₀ the decayed stored stresses and K = Σ E r dz over every
slice; XPBD solves it as a spring with stiffness factor K/(E·h_ref) about a shifted
zero-force point, with λ clamped to the depth-integrated caps (h_ref =
`stiffness_reference_km` 150, the thickest column, so each factor is ≤ 1). A cracked
pair is therefore pulled together at most at its tension cap until it reaches
touching. The tangential row is the same with G_t, its zero point an offset; friction
caps use the start-of-step normal stresses, so they are fixed within a step. The
caps are applied per slice only after the solve: within a step, the total can sit
under its cap while one slice is over its own, and the next step corrects it.

**Limits of the continuum, by state, not by kind.** φ = 0 between cold, shallow
columns: tensile cap ≈ 0, a contact. High φ: a weld. Low φ under shear: a fault or
transform. Hot, thin, stretching columns: low caps everywhere, a ridge. A broken edge
with a bending flap on one side: a subduction interface (section 6).

**What the law gives** (CPU reference, default parameters): an old-ocean pair (100
Myr, 100 km shared) has compression cap 11 400 MPa·km and tension cap 4790 MPa·km at
φ = 1, of which the brittle slices' frictional extension strength is 3110. At φ = 0
its tension cap is 1680 MPa·km, the ductile base's flow stress, about 80% of ridge
push (~2100 MPa·km at 50 Myr): a broken
boundary between old plates resists opening nearly as hard as an intact one unless
`ductile_damage_weakening` is raised (section 10). A fully cold pair at φ = 0 has
tension cap exactly 0.

**Checked:** under constant separation rate the stress follows the implicit-Euler
Maxwell sequence to 10⁻¹² (CPU) and 4 digits (GPU); a cold pair holds its stress, a
hot pair's is gone a step later; a cold φ = 0 pair carries no tension and closes
exactly at touching; the kernels match the reference on six real-profile pairs to
≤ 10⁻⁶. Two broken cold pairs turned against each other (one pressed by 10⁻³) follow
the reference slice by slice to 2.4×10⁻⁶ of the cap through the elastic rise to the
friction plateau, the pressed pair's caps higher by μ × its normal stress. Rolling:
stiffness and kern cap exact.

### 4.6 Gravitational potential energy (ridge push, plateau collapse)

A thin-sheet column model loses the horizontal force from lateral differences in
gravitational potential energy (GPE). It is added as an explicit body force:

> Fᵢ = −Aᵢ ∇Γ

The gradient is a least-squares fit over the column's listed neighbours (start-of-step
positions), applied after the edge update as a shift of the predicted position,
Δx = −∇Γ·h·dt/η_a (against the column's basal drag; independent of its area;
`columns_gpe_force` in `flow.wgsl`, reference `flow.js`). With depth–age cooling this
is ridge push; on thick continents, plateau collapse and orogenic spreading.

`gpe_force_factor` is **0 by default**. The force is physical (ΔΓ ≈ 2×10¹² N/m across
a continental margin is realistic), but basal drag at η_a = 10¹⁹ Pa·s over a ~200 km
channel is too weak to hold it: a free column at a ridge-like age gradient (9–51 Myr
across 900 km) is pushed at 141 cm/yr, and on a generated planet margin columns move
at metres per year and edges break. Turning it on is part of calibration (section 9.2).

### 4.7 Irreversible processes (between solves)

All are applied once per step, after the solve, from converged stresses
(`edges_maxwell`, `columns_flow`). This keeps the solve linear within a step and
means solver residuals below yield (~MPa) change nothing permanent.

- **Edge memory is the pair's geometry** at the last update (length and direction).
  The rest length then only sets where new edges start, where columns are in range,
  and the strain scale. Changing it creates no stress, so area and shape changes are
  outlets, not double counts.
- **Inelastic strain** of an edge in a step: the strain increment its *stiffest*
  slice (largest Maxwell factor) did not store elastically. A column deforms
  permanently only as fast as its strongest layer lets it:
  - a cold lid below yield holds the column however much the hot base creeps (stress
    relaxing below is load moving to the lid, not shortening);
  - a lid at its cap lets the column flow; an all-hot column flows by its coldest
    part;
  - crack opening is not inelastic: it closes again;
  - an edge in its first step contributes none: what its slices relax is the load it
    was born with, not flow (in hot rock, the gaps of new neighbours would otherwise
    read as dilatation, e.g. after neighbour exchange in shear).
- **Volumetric flow:** d ln area_ref = the trace of the column's fitted inelastic
  strain (4.2), × `volumetric_flow_factor` (1). Crust volume is unchanged, so
  thickness (V / cell area) follows the geometry, and isostasy moves the elevation.
  Where the tensor cannot be fitted (edges in fewer than two directions), 2 × the
  mean of the edges' inelastic strain if there are at least three edges, else
  nothing; the mean is the trace only for edges spread evenly in angle. In pure shear
  nothing changes area. The GPE term reaches it through the GPE force (4.6) loading
  the edges. This is the thin viscous sheet of England & McKenzie (1982), whose
  Argand number (buoyancy stress / viscous stress) decides between thickening and
  collapse.
- **Shape:** the deviatoric part of the fit, turned with the material (4.2).
- **Plastic work:** cap × plastic strain × dz, summed over slices at a cap in both
  channels, plus cohesion × opening for cracks. Creep below a cap is not counted.
- **Integrity:**

  > φ ← max(φ − W/W_D, 0), then 1 − φ decays by exp(−dt/τ_heal(T))

  W is positive for compression, extension and shear alike, so any permanent
  deformation reduces integrity. τ_heal = `heal_time_myr` × exp(Q_h/R (1/T − 1/T_m)),
  T the temperature of the edge's load-bearing part (slice temperatures weighted by
  r·dz): hot material heals fast. Defaults: W_D = 50 MPa·km; τ_heal = 0.1 Myr at
  mantle temperature, 53 Myr at 1200 K, effectively never below ~900 K. This mirrors
  grain-damage theory of plate-boundary formation (Bercovici & Ricard 2012); W_D and
  τ_heal decide whether damage localises into narrow boundaries (section 10).
- **Thermal:** thermal age advances (capped by composition); stretching scales it
  (4.8); arcs reset it (section 6).
- **State transitions:** suturing via the edge timer; subduction when a trench hinge
  passes a column's back face (section 6).

**Checked:** uniform shortening at the caps thickens a 19-column patch at the
predicted rate (ln 0.0479 over steps 6–12 against 0.0481) with volume held to 10⁻⁵;
integrity matches 1 − ΣW/W_D to 2×10⁻⁷; pure shear changes no area; a patch below
yield, solved for 30 steps, does zero work and keeps integrity exactly 1.

### 4.8 Area, volume, melt and splits

**Area is geometric** (`columns_area` in `area.wgsl`; reference `src/columns/area.js`).
A column's actual area is its cell in a weighted Voronoi tessellation of the column
positions. Between columns i and j the boundary is the great circle in the plane
x·(aⱼpⱼ − aᵢpᵢ) = 0, a = 1/cos(apothem):
- between touching columns it passes exactly through the contact point; between
  equal columns it is the perpendicular bisector;
- the normals of any three such planes sum to zero, so the planes share a line and
  the cells meet at common vertices: the cells tile the sphere exactly, whatever the
  weights (a spherical cousin of the power diagram).

Each cell is computed alone, in the gnomonic projection about its column, where great
circles are straight lines: a regular 12-gon of radius `cell_clip_radius` (0.04)
clipped by one half-plane per candidate (Sutherland–Hodgman), its spherical area
summed over fan triangles (Van Oosterom–Strackee). Everything is in the column's own
tangent coordinates, so float32 is enough. Candidates are all columns in the 27 hash
cells around (not the listed edges, which a column at an opening gap would lose); the
clip radius is kept below half a hash cell, so every column whose boundary can cut the
clip polygon is seen. A cell still touching its clip polygon has a free side: there
the tiling is not exact; such cells are counted (`clipped_cells`) and never split.
What would be a void is a column whose cell is stretched: it thins, subsides, floods
and melts. Passive rifting needs no special rule.

**Volume and thickness** (`columns_volume`; reference `src/columns/volume.js`).
Crust volume V is state; thickness is V / cell area, every step. At load, V = loaded
thickness × cell area. Per step, from last step's cell area A₀ to this step's A:
- **Thinning:** thermal age t ← min(t, cap)·(A₀/A)², which thins the lithosphere by
  A₀/A, McKenzie's (1978) uniform stretching factor β. It recovers by ageing: a
  continent re-thickens to 150 km in ~130 Myr. (The clamp comes first: a continent
  loaded with age 1000 Myr would otherwise stretch for many steps before thinning.)
- **Decompression melt** (McKenzie & Bickle 1988):
  dV = F·H_m·max(A − A₀, 0)·max(1 − L/H_m, 0), L the thinned lithosphere,
  H_m = `melt_depth_km` 70 (where rising mantle crosses its solidus), F =
  `melt_fraction` 0.1. F·H_m = 7 km is why steady ridges make 7 km crust at any
  spreading rate. Under thick lithosphere stretching only thins; melt begins once it
  is thin enough.
- **Composition:** melt (`melt_density` 3000) is mixed in by volume. A continental
  column crossing 2850 kg/m³ this way has broken up: it is counted (`breakups`) and its
  formation age reset. This happens only at β ≈ 10 (section 10).
- **Arcs** add volume above slabs (section 6).
- **Sinks:** columns becoming slabs (their volume leaves the surface budget).
- **Budget:** sources and sinks are summed on the GPU in 64-bit fixed point
  (10¹² units per km·unit²) since load, so Δ(Σ V) = sources − sinks can be checked
  exactly.

**Splits** (`columns_split`, `crust_commit_splits`, `crust_rewire_edges` in
`split.wgsl`). A surface column whose cell is not clipped splits, at the start of a
step, on last step's cells:
- **By shape** (first): when its reference shape is elongated past
  `split_aspect_ratio` 2 and its reference area is at least `split_area_guard` 1.3 ×
  the base hexagon. It splits across its long axis: each half has half the area and
  half the aspect ratio (round at 2; axes swap below), and sits at its own radius
  along the axis from the parent's position, so the halves touch at their rest
  length. A ridge stretches columns along the pull, so they split into rows parallel
  to the ridge; elongation at constant area (shear) never splits.
- **By area** (the backstop, for stretching with no preferred direction): when its
  actual cell area exceeds `split_area_factor` 2 × the base hexagon. The halves go to
  the cell centroid ∓ 0.87 × the cell's major-axis spread (√ of its largest second
  moment): the two half-cells' centroids for a domino of hexagons (0.885), a rectangle
  (0.866) or an ellipse (0.849). The shape is kept.

Either way: parent and child share volume and area_ref equally; the child copies the
parent's material (it is not fresh oceanic crust); the edge between them starts
intact and stress-free; edges to former neighbours are re-attached to whichever half
is closer, keeping their integrity and stored stresses, and every edge touching
either half restarts its length and direction from the new geometry, so a split adds
no stress; neither half counts dilatation in the step after the split. Slots are
allocated in two phases (tickets against both free lists, then one pop), so the free
lists never change while being read.

**Checked:** cells sum to 4π to 5×10⁻¹⁴ (CPU) on lattices and jittered configurations
with apothems ±40%, and to 10⁻⁸ relative on a 2562-column GPU sphere, each cell
matching the reference to 7×10⁻⁷; stretched young columns make 7.0000 km of crust
per unit new area, old ocean and continents none; the budget closes to −3×10⁻⁸
against a float32 rounding bound of 10⁻⁶; 342 area splits keep Σ V and Σ area_ref,
with halves at the reference positions to 5×10⁻⁵ spacings and the tiling exact to
4×10⁻⁸ afterwards. Hot columns stretched by hand split at stretch 2.1 into touching
round halves (aspect ≤ 1.015) lined up along the pull, and Σ V = initial + melt.

---

## 5. The time step

### 5.1 Schedule

One step (`Simulation.step` in `src/sim/simulation.js`); every kernel reads and
writes named buffers only, so any one can be run alone on dumped or synthetic inputs:

```
 1. locate          nearest mantle node per particle (downhill walk)
 2. splits          by shape or area, on last step's cells; commit slots; re-attach edges (4.8)
 3. predict         x̃ = x + v_mantle·dt; spin = 0; hash grid of predicted positions
 4. columns         tessellated area; dilatation: thinning, melt, volume, thickness (4.8);
                    profiles (4.0); ages; slabs warm, hang or sink, neck, supply arcs, are
                    removed (6);
                    arc growth (6)
 5. bending         elevations: Winkler isostasy with plate bending, Jacobi sweeps (4.3)
 6. edges           prune pairs out of range, create pairs in range, sort slot lists (3.2);
                    GPE push into the predicted positions (4.6); trench hinges: bend,
                    subduction, slab pull (6); each edge's linear law, caps, rest length,
                    lever, face (4.5)
 7. solve           warm start (positions to last step's edge forces); solver_iterations
                    (32) sweeps: all edges' normal, tangential and rolling rows (Jacobi),
                    Chebyshev extrapolation of the normal multipliers (5.2)
 7b. no net rotation  the area-weighted rigid rotation of the step removed (5.3)
 8. irreversible    slice stresses, inelastic strain, slip, plastic work, integrity, moment,
                    edge timers and suturing (4.7, 6); area and shape (4.7)
 9. integrate       position = solved position; velocity (diagnostic); body axis carried and
                    turned; columns marked for subduction become slabs (6)
CPU, between steps: mantle advanced and uploaded every stepMyr; plate labelling every
0.25 Myr (3.3); recording when due (8)
```

**Problem size.** Buffers are allocated for `CAPACITY` (25,000 columns, 131,072 edge
slots), but GPU work is sized to the slots in use. The free lists hand out the lowest
free slot first, and the GPU keeps high-water marks (one past the highest particle
and edge slot ever used; `note_particle_slot`, `note_edge_slot`). Per-particle and
per-edge kernels are indirect dispatches over those marks (`solver_dispatch_sizes`,
refreshed after splits and after new edges), draws are indirect over the particle
mark, and snapshots read back only the slots in use. A load or restore rebuilds the
marks with one full scan (`solver_find_high`). (Before, every kernel launched
threads for every slot and every frame drew 25,000 instances: a 162-column world
cost what a 10,000-column one did, and recording read 35 MB back every 1 Myr.)

Steps with many iterations are submitted in pieces, waiting for each, so the
browser's GPU watchdog does not kill the device: a piece holds about 10⁶ thread
launches (from the high-water marks, read back a step late), at least 16 solver or 64
bending iterations, the pieces that were always safe at capacity. A 162-column world
does its whole step in one piece. `runSteps` keeps at most two steps in flight.

### 5.2 XPBD with drag in place of mass

Multiplying Φ by dt gives ½|x − x̃|²_Γ + dt·U. For a constraint Cₖ with compliance αₖ
(energy Cₖ²/2αₖ):

> Δλₖ = (−Cₖ − α̃ₖ λₖ) / (Σᵢ wᵢ |∇ᵢCₖ|² + α̃ₖ),   Δxᵢ = wᵢ ∇ᵢCₖ Δλₖ
> with **wᵢ = 1/γᵢ** (instead of 1/mᵢ) and **α̃ₖ = αₖ/dt** (instead of αₖ/dt²)

Spins enter the same way, with weight 1/(spin drag). Velocity is not state; it is a
diagnostic. Large dt is safe; it only makes each step more nearly quasi-static.

**Jacobi, scaled.** All edges are updated simultaneously; each Δλ is scaled by
1/(edges at either end), and the scaled Δλ is accumulated in full, so λ stays
consistent with the displacement it caused and a converged solve is the true minimum
of drag + edge energy. Corrections are summed with fixed-point integer atomics (WGSL
has no float atomics). Graph-coloured Gauss–Seidel would gain ~2× in rate but needs
its colouring maintained as splits and rewiring change topology, and does not change
the (L/ℓ)² scaling that is the actual problem.

**Chebyshev acceleration** (Wang 2015) of the normal multipliers: λ_{k+1} =
ω_k(λ̂ − λ_{k−1}) + λ_{k−1}, λ̂ one plain sweep from λ_k, clamped to the caps, after 3
plain sweeps; each particle moves by exactly the change applied, so x − x̃ = W Bᵀλ
holds at every iteration, clamped or not. (Until the redesign below, the positions
were extrapolated separately with the same weights; whenever a cap clamped the
extrapolated λ the two parted, motion without a force, which the momentum then
amplified: the "divergence under concentrated pull" of section 9.2.) The sweep acts
on the multipliers as I − (BWBᵀ + α̃)/(d(w_a + w_b + α̃)), B the edge–particle
incidence; its spectrum lies in [0, 1). Self-equilibrated force patterns (null space
of Bᵀ, the loops of a 2D lattice) are slowest, ρ = 1 − α̃/(d(2 + α̃)), α̃ = 1/(κ·dt),
d = 6: 0.99993 at 70 GPa and dt = 16 kyr. On a 61-particle pre-compressed chain at
κ·dt = 200, after 128 sweeps the error is 0.4% with Chebyshev and 76% with plain
Jacobi.

**Warm start.** Each edge's multiplier starts at the force it applied in the previous
step's solve (kept in λ between steps as MPa·km, within the new caps), and the
positions move to match, x₀ = x̃ + W Bᵀλ₀: the solve starts from last step's motion
relative to the mantle and corrects the increment, instead of rebuilding every
stored stress from zero, which is exactly the slow, self-equilibrated part. Edges
that are new, or whose pair was just split or became a slab, start at zero. (Not
from the stored stress: moving the positions by that is an explicit Euler step of the
stiff edges, κ·dt ≈ 250 here, which amplified whatever a short solve left: at 32
sweeps edge forces grew to 100× the drag within 2 Myr.)

**Choosing ρ (`chebyshev_rho` < 0, `chebyshevRho`).** ρ = min(derived, 1 −
½(4/N)², 0.999), N the accelerated sweeps. Two findings:
- *A budget can only use the ρ it can finish.* After N sweeps Chebyshev bounds every
  mode in [−ρ, ρ] by 1/T_N(1/ρ) ≈ 1/cosh(N√(2(1−ρ))). With the derived 0.99999 and
  N = 29 that is 0.99 for *every* mode, fast ones included, which plain Jacobi
  removes in a sweep: the accelerated solve was worse than plain Jacobi. The cap puts
  N√(2(1−ρ)) = 4 (bound 4%); slower modes are left to the warm start.
- *Overestimating ρ is harmless for a linear solve, not with caps.* A strip under
  concentrated slab pull, 256 sweeps: exact at ρ = 0.999, a column torn ~280 km at
  0.99987. Hence the ceiling. (Measuring ρ each step instead, from the contraction of
  the last plain sweeps, gave ~0.8: it sees the fast modes and underestimates, and
  tripled the velocity error on the two-plate planet.)

Measured on the two-plate planet (one step from the same state, against a
1,024-sweep solve; plate speeds ~2.9 cm/yr rms):

| sweeps | start | ρ | velocity error | stress error (rms / max) | time (software GPU) |
|---|---|---|---|---|---|
| 256 | cold | 0.999 | 2.1% | 0.15 / 0.5 MPa | 1.23 s |
| 32 | cold | 0.992 | 31% | 3.2 / 9.3 MPa | 0.30 s |
| 32 | warm | 0.999 | 6.2% | 1.5 / 5.5 MPa | 0.29 s |
| 32 | warm | 0.992 (rule) | 5.8% | 0.28 / 0.8 MPa | 0.29 s |
| 32 | warm | 0 (plain) | 6.5% | 0.69 / 2.6 MPa | 0.27 s |
| 64 | warm | 0.998 (rule) | 4.6% | 0.23 / 0.7 MPa | 0.47 s |
| 16 | warm | 0.969 (rule) | 7.3% | 0.54 / 2.0 MPa | 0.16 s |

At 32 sweeps any ρ in 0.97–0.992 does the same. Over 30 Myr the warm 32-sweep solve
keeps the median residual force (edges + drag, which omits the GPE push and slab pull)
at 1.4× the cold 256-sweep solve's (20/27/38 vs 13/19/28 at 10/20/30 Myr; the rise is
the trench developing, in both); 64 sweeps are no better than 32.
- **Linear within a step.** Anything that changes the operator inside the iteration
  (damage, stiffness switching with the sign of the stress) made accelerated runs
  diverge; such changes happen between solves. The λ clamps (caps, contacts) are the
  remaining nonlinearity (ceiling on ρ, above).
- **Tangential and rolling rows are plain Jacobi.** For ρ near 1 the Chebyshev
  weights converge every mode at the slowest rate, and spins, which plain Jacobi
  converges in a few sweeps, were left at ~4% of the naive shear. So their position
  corrections go to a separate accumulator, applied with the normal rows' change but
  not extrapolated, and their multipliers and the spins are not extrapolated.
- **Branching histories.** The trench hinge strip (section 9.2) is not a convergence
  test: from t = 0.5 Myr, solves that differ by little (ρ 0.999 or 0.9992 at 256
  sweeps) land, 2 steps later, on one of two self-consistent states (strip speed 31
  or 48 cm/yr), each reproduced by several settings. Its test pins a converged
  solve (256 sweeps, ρ 0.999).

**No coarse plate-level solve.** Positions always satisfy x − x̃ = W Bᵀλ, so every
plate's drag-weighted net motion is already exact at every iteration; what converges
slowly is the distribution of force within a plate (the multipliers). Moving
positions rigidly without a matching λ would change the physics, and a matching λ
needs the global solve being accelerated. Alternatives, if iterations become too
costly at fine resolution, are in section 9.3.

### 5.3 No net rotation

Each step, after the solve and before the irreversible updates, the planet's net
rotation is removed (`remove_net_rotation`, on by default; `crust_net_rotation_*` in
`solver.wgsl`, CPU reference `netRotation` in `src/geometry/net-rotation.js`). The
net rotation is the rotation vector θ that best explains the step's displacements
u = x̃ − x of the surface columns as one rigid rotation, weighted by cell area:
it minimises Σ a |u − θ × x|², so I θ = L with L = Σ a x × u and I = Σ a (1 − x xᵀ).
On a fully covered sphere I = (2/3)·4π·1, and θ is the area-mean of x × u. Every
column, slabs included, is then turned by −θ, and its spin loses θ·x, so no
length and no face sliding changes. Velocities, Euler poles and arrows are therefore
in the no-net-rotation (NNR) frame that geodesy uses for plate motions.

- GPU: a shared-memory reduction per workgroup, one float atomic per workgroup and
  sum, a one-thread 3×3 solve. A patch too small to define a rotation (I nearly
  singular) removes nothing. Small angles use series for sin and cos: WGSL allows
  them ~5×10⁻⁴ absolute error, more than a step's whole rotation (the first version
  misplaced columns by 3×10⁻⁵ rad that way).
- Turning by −θ removes θ to first order; what is left is ~|θ|·|u|, 10⁻⁴ of θ in
  the test's 0.03 rad step and far less at real step sizes (~10⁻⁴ rad).
- It is a frame change for the crust only: the mantle is not turned with it, so the
  next step's drag is computed against the mantle where it is. In effect the crust
  gets whatever uniform torque cancels its net rotation, and stays registered to
  the mantle's pattern instead of drifting over it. A lava-lamp mantle has no net
  rotation of its own (a gradient flow), so there the removed part comes from the
  crust: drag varying with lithosphere thickness, and slab and GPE forces.
- Mechanism tests on an empty page turn it off: on a patch it would remove the
  patch's own rigid motion.

Measured (two-plate world, software GPU): the removed rotation is a steady
0.34°/Myr about the mantle's Euler pole, half the mantle's 0.72°/Myr, and what is
left is 10⁻⁵ of it. Without the removal both plates turn together at ~0.33°/Myr
(each half's fit about the pole: 0.31–0.37°/Myr over 1.5 Myr) — the trench is
locked (9.2), and the two plates move as one shell dragged by half the mantle. With
it, each plate's rotation about the pole is under 0.04°/Myr and the rms speed
0.3–0.9 cm/yr: the net rotation was nearly all of the motion.

### 5.4 Parallelism and determinism

- Jacobi sweeps with per-iteration integer atomics; no graph colouring.
- Thread-timing-dependent behaviour (free-list pops, edge creation order, which free
  slot a splitting particle receives) is listed in record manifests, not engineered
  away. Runs agree statistically, not bit for bit; a run resumed from a record in a
  fresh page continues it to 5×10⁻⁵ on the unit sphere (spacing 0.035).

---

## 6. Boundaries, subduction and spreading

Under the pair law boundaries are not built; they are where edge state and column
profiles have become weak. Kernels: the trench hinge in `hinge.wgsl` (reference
`src/columns/hinge.js`); the edge timer and suturing at the end of `edges_maxwell`
(`src/edges/transitions.js`); slabs in `thermal.wgsl`; arcs in `flexure.wgsl`.

**Rifting and spreading.** Where columns are pulled apart, tension is resisted by the
ductile slices whatever the integrity, so separation is taken up by stretching:
cells grow, crust thins at fixed volume, the column subsides and, below sea level,
floods. Stretching thins the lithosphere, and hot is weak, so strain localises: the
positive feedback by which real rifts narrow. Once the lithosphere is thinner than the
melting depth, new area arrives with melt (4.8) and the column becomes oceanic:
breakup. Stretched columns elongate along the pull and split into rows across it;
the halves become new sea floor (4.8). Ridges stand high because young lithosphere is
thin and hot (4.4), and ridge push follows (4.6, when on). Arc heat also makes melt
where a reset column's cell grows, so arcs and rifting interact.

**Shear and transforms.** Shear loads edges tangentially. In brittle rock it builds
to the friction cap and slips (a fault), with plastic work lowering integrity; in
ductile rock it flows and elongates the columns' shapes, and past the aspect cap
pairs leave range and others enter (neighbour exchange). Low-integrity edges under
sustained shear are transforms and strike-slip faults.

**Subduction: the trench hinge.** A plate goes down at a bend inside a tile, not by a
stress threshold. Kernel `columns_hinge` (`hinge.wgsl`), after the edges are listed
and before the edge law; reference `src/columns/hinge.js`.
- **Trench partner.** A surface column that can sink (freeboard < 0) faces the more
  buoyant surface neighbour across a broken edge (integrity < 0.5) that it is furthest
  under; equal columns have no polarity. Failing that, a slab hanging from it.
  Freeboard < 0 makes ocean older than ~8 Myr subductable (+0.30 km at 2 Myr, −0.14 at
  10, −1.27 at 60; a continent +3.41): basaltic crust's compositional buoyancy is
  small at 3000 kg/m³ (section 10). Buoyant columns never go down.
- **Where the hinge is.** A plate on a buoyant foundation loaded at its end bends
  within its flexural length ℓ = (4D/K)^¼ of the end, and its moment peaks at
  x* = πℓ/4 behind it (Hetényi): there is the hinge, behind the trench line. The
  trench line is the overriding partner's front face, so u, the part of the column
  already past it, is the pair's map-view overlap. Flap length L_f = x* + u, tip
  depth L_f sin β. Rotational stiffness of the hinge D/ℓ. For 30 km of elastic
  thickness ℓ ≈ 68 km, x* ≈ 53 km, D/ℓ ≈ 2.5×10¹⁸ N·m/m per rad.
- **Bend β, quasi-static each step** (it relaxes on the isostatic time, ~30 kyr),
  per unit hinge length: drive = F_s L_f + w_u(L_f² − x*²)/2 (F_s what hangs from the
  tip, w_u the negative buoyancy per area of the underthrust part, which is immersed
  in mantle, less the viscous support of its descent); resistance = (K x*³/3 − P L_f)β
  + M(β) (the foundation under the outboard part; trench compression P destabilises a
  dipping flap; M the hinge moment, elastic up to the plastic moment M_y, the
  column's integrated slice strength about its strength-weighted median depth). Where
  the net stiffness is negative the bend runs away to `hinge_max_angle` (1 rad). With
  no slab and no underthrust, nothing bends: compression alone buckles a plate only
  above ~K ℓ²/2 ≈ 10¹⁴ N/m, a hundred times what drag supplies. Spontaneous
  initiation is hard, as in nature (McKenzie 1977; Mueller & Phillips 1991).
- **Contact.** The trench pair's compression cap goes from the face strength to the
  lubricated interface (`slab_interface_weakening`) as β reaches the optimal thrust
  dip θ* = ½ atan(1/μ) (29° at μ = 0.6): a vertical face cannot slip under horizontal
  push, and Coulomb resistance is least at θ*. Underthrust is not compaction: the pair
  feeds no volumetric flow, and an underthrusting column's thickness is over its
  reference area, not its shrinking cell.
- **Becoming a slab:** when the hinge passes the column's back face, u + x* ≥ 2a
  (the whole column is flap). There is no timer and no propagation rule: a trench
  propagates where neighbours have partners and are pulled.
- **Slab pull** reaches the plate as a body force on the hinge column toward the
  trench (into its drag target, like the GPE push): the down-dip part, sin β, of what
  hangs from the flap plus the underthrust part's weight. At most what the column can
  transmit (its tensile strength across its width) and what the plate behind can hold
  (the tension caps of its edges behind, projected on the pull, × 0.9); more would
  tear the plate behind the hinge.

**Slabs hang.** A new slab hangs from the intact surface column behind it (found by
the hinge), at its bend as dip; a slab can hang from a slab (a chain). Kernel
`slab_hang` in `thermal.wgsl`; reference `hangSlab`.
- **Where:** in map view at the trench (its parent's front face toward the parent's
  trench partner, or under its parent slab); its top at the parent's flap tip, or
  the parent slab's bottom (top − extent, extent = 2a sin dip). It has no edges.
- **Load:** its weight above the cutoff (B g A, B = −ρ_m × freeboard), less the
  viscous support of its descent (descent rate = the parent's speed toward its
  trench × sin β, over the free sinking rate), plus what hangs from it; under-relaxed
  (half the change per step: support, descent and bend depend on each other a step
  apart, and unrelaxed the lagged fixed point flips). Loads accumulate upward by
  atomic float adds and reach the parent next step.
- **Necking and break-off:** at most the tensile strength of the weaker column × the
  narrower width passes up (`slab_attach_factor` 1; 0 = slabs never hang). The excess
  necks the slab: its top falls behind the hang point at w_free(1 − cap/load). Necked
  through its thickness, it breaks off and sinks freely.
- **Free sinking:** w = B g h/4η, h the slab's thickness (its column extent): a sheet
  sinks edge-first, whatever its width in map view (~3 cm/yr for old ocean). (It was
  B g (2a)/4η, which made a 1,900 km column sink at 50 cm/yr.)
- **Free slabs** keep their 3D-measured edges and the old rules: an edge with a slab
  end is measured in 3D, so a sinking slab stretches it and pulls the plate behind
  it, until the depth intervals stop overlapping (~100 km down).
- **Warming** (thermal age × exp(−0.05 dt)) and **removal** at `slab_cutoff_km` (660)
  or neutral buoyancy, as before. A slab lost (broken off or removed) while hanging
  from a surface column is counted (`slabsLost`): the heuristic *a slab does not
  disappear before the tile it drags has started to subduct* as a number.
- **Weak interface:** compression caps on edges with exactly one slab end × 0.1. A
  slab's overburden in its profile is mantle rock, not water.

**Arcs.** A slab crossing the arc window (80–160 km; a free slab's top, a hanging
slab's bottom) supplies `arc_efficiency` (0.2) × its volume over the passage, shared
equally among surface columns within `arc_reach` (0.05, ~320 km) on the far side from
the plate behind the slab (what it hangs from; for a free slab the integrity-weighted
direction to its surface edge partners). It is mixed in at
`arc_density` 2750 (so arcs build continental crust) and logged as a source; the
receiving columns' thermal age resets to 0 (hot, weak arc lithosphere). Arcs feed
only the far side: feeding the slab's own plate reset its lithosphere hot, which ended
the depth overlap and detached the slab.

**Collision and suturing.** Buoyant columns in sustained compression thicken through
volumetric flow (4.7), bounded by GPE spreading. If both columns are buoyant, the
timer passes `suture_window_myr` (5 Myr) and the edge did no plastic work that step,
the edge welds (integrity 1); the plate labelling records the merge.

**Checked:** the hinge's setback and stiffness against a finite-difference plate on a
foundation (peak moment within one grid step of πℓ/4; end stiffness within 2% of
D/ℓ); the elastic, plastic and runaway branches and partial unloading. On the GPU,
compression alone (old ocean held against a continent at 0.2%) subducts nothing, and
two buoyant pairs weld at their window. A strip (eight old-ocean columns against a
four-column continent, a slab hanging from the leading column, no mantle, damage off,
dt 0.25 Myr, a converged solve): the flap bends 3.6° in the second step, the leading
column goes up to 118 km under the continent and subducts at 1.5 Myr, its slab still
hanging from it (5.25 Myr and 145 km before positions followed the multipliers, 5.2);
no slab lost, no continent subducted, its volume logged as the sink. A free slab sinks
0.52 km per step (3.2 cm/yr, the reference to 5×10⁻⁷), pulls the plate behind it and
is removed at the cutoff; the arc gets exactly 20% of its volume, all on the far side.

**Coarse resolution (~220 km).** Slabs are one column, bending is aliased, and arcs
are one column wide. The same rules run; slab pull is lumpier. Deviation from Earth
statistics is accepted.

---

## 7. Resolution

- **Base resolution** is set by `base_apothem` (0.0175 on the unit sphere, ~110 km,
  so ~220 km spacing and ~10k columns). Continuum parameters are converted per column
  and per edge from physical values (principle 5): edge stiffness per unit modulus is
  independent of spacing, drag scales with area, so the stiffness-to-drag ratio κ
  scales as 1/spacing².
- **Consequences at finer spacing:** κ·dt is ~19× larger at 50 km, Chebyshev
  iterations grow as √(κ·dt) (~4.4×), particle count 22×; bending sweeps grow as
  (α/ℓ)⁴. The cost per simulated Myr scales as 1/√dt (longer steps are cheaper, up to
  where per-step topology changes need finer time resolution). These are to be
  measured before the solver changes (section 9.2).
- **Hash grid:** uniform over [−1.5, 1.5]³, `hash_grid_dim` 32 cells per axis
  (`?grid=N` overrides it for coarse test worlds). The cell must exceed the largest
  contact distance, and the tessellation's clip radius must stay below half a cell.
- **Precision:** float32 on the unit sphere resolves ~0.4 m. Lower levels use local
  coordinates (section 9.4).

---

## 8. Records and contracts between scales

### 8.1 Vocabulary and rules

- **Level:** one model at one scale, e.g. L0 mantle (optional), **L1 global plates
  (this model)**, L2 regional crust patch, L3 local terrain.
- **Record:** what a run writes, self-describing and versioned.
- **Forcing:** what a run reads.
- **Adapter:** turns a parent's record into a child's forcing (cropping,
  reprojection, resampling); the only code that knows about two levels.
- **Generator:** produces a forcing procedurally, with no parent run, in the same
  schema an adapter would, so a child cannot tell them apart. The lava-lamp mantle is
  L1's generator of basal forcing (3.4).

Rules:
1. **One-way.** Children never alter parents. Disagreement is measured and reported,
   never fed back.
2. **Fields, not particles.** Children consume Eulerian fields sampled on a stated
   mesh, so they are independent of the parent's discretisation. Lagrangian snapshots
   exist for restart and debugging; no child depends on them.
3. **Every quantity declares** its units, reference frame, sampling (point value vs
   cell average) and extensiveness. Extensive quantities are remapped conservatively;
   intensive ones are interpolated.
4. **Time resolution is part of the contract.** A record states its sampling interval;
   a child must not depend on parent variability faster than that.
5. **Records are traceable.** The manifest names the code, seed and parameters, so the
   run can be regenerated to statistical (not bitwise) agreement.
6. **Policy for disagreement:** the parent is authoritative at the child's lateral
   boundary, the child in its interior, with a relaxation zone between them (as in the
   lateral boundary conditions of regional climate models, "dynamical downscaling").

### 8.2 Frames and time

- **Global frame:** unit sphere, body-fixed (planetary rotation is not modelled).
  Latitude/longitude derived from it; conventions stated in the manifest.
- **Time:** Myr since the start of the run that produced the record.
- **Vertical:** heights in km relative to sea level.

### 8.3 The record (what this level writes)

`src/record/`; schema `plate-player-record` 0.7.0. Every run is recorded unless
switched off, and downloads as one ZIP.

**a. Manifest** (always): schema version, model name, build (git commit), seed and
initial-state options, full parameter set, resolution, frames, time range, the
included tiers and fields, the struct layouts, known nondeterminism, and where a run
was restarted from.

**b. Field series:** every `fieldIntervalMyr` (1 Myr) on a fixed icosphere (5
subdivisions, 10 242 nodes, ~220 km, matching base particles). Particle quantities
are sampled by a Gaussian kernel average (weight = cell area × exp(−d²/2σ²), σ = one
base spacing, cutoff 3σ); slabs contribute only to `slab_fraction`; plate ids are
nearest-particle (ids are not averaged). Fields: `surface_height`,
`crust_thickness`, `crust_density`, `crust_age`, `temperature`, `velocity_east`,
`velocity_north`, `edge_integrity`, `shape_aspect`, `vertical_axis_rotation`,
`slab_fraction`, `mantle_velocity_east`, `mantle_velocity_north`, `mantle_buoyancy`,
`plate_id`, `support` (coverage). Each declares units, long name and kind
(intensive, mantle, categorical, diagnostic) in CF style, so NetCDF or Zarr
conversion is a renaming. One little-endian binary array per field, shapes and dtypes
in the manifest.

**c. Snapshots:** the complete raw GPU state (particles, edges, free lists, mantle
grid) every `snapshotIntervalMyr` (25 Myr), plus the CPU mantle state; enough to
resume a run in a fresh page.

**d. Events and plates:** `events.json` (`plate_split`, `plate_merge`, `plate_lost`)
and a plate time series (`plates/plates.json`: per plate its size, Euler pole and rate,
rms and non-rigid speed, and a rigidity residual rms(v − ω×r)/rms(v)). Counters since load (volume
sources and sinks, breakups, clipped cells, capacity overflows) are readable from the
solver buffer.

### 8.4 Downward to the game

The game is just another consumer. Height maps, biome-relevant fields (crust age,
plate boundaries, volcanic arcs) and the event log are rendered from the same
records. Nothing game-specific lives in the simulation.

---

## 9. Development plan and extensions

### 9.1 Development strategy: assemble, then run

**Why.** The phenomena available for calibration are coupled: plate speed depends on
basal drag, edge strength, slab pull and ridge push at once; rigidity on solver
convergence and on yield; ridge depth on cooling, isostasy and melt. Tuning one
component against such observables while the others were missing or stand-ins
repeatedly moved the problem rather than solved it. So:

1. **Components are developed against known answers.** Each is checked against an
   analytic solution, a conservation identity, a limit case, or a reference
   implementation, not against planet behaviour.
2. **Mechanism tests stay cheap.** Under the software GPU buffers, scans at load and
   full readbacks cost in proportion to capacity, and a readback takes ~0.5 s. Mechanism
   tests run on an empty page with small buffers (`?empty&capacity=1024`), write
   inputs without readbacks, and read back as rarely as they can; the default GPU
   suite takes ~6 minutes. Tests on a generated planet run only with
   `PLATE_PLANET=1`.
3. **Every component is inspectable on its own.**
   - A pure CPU reference in JavaScript (plain arrays to plain arrays) next to each
     WGSL kernel; unit tests check the reference, GPU tests the kernel against it.
   - Each kernel reads and writes named buffers, so a suspected fault can be tested on
     harvested or synthetic data without warming up a planet (README, "Debugging").
   - Intermediate per-column and per-edge quantities (profiles, stiffness, caps, φ,
     plastic work, melt, tessellated area, shapes) are exposed through a debug
     readback (`plate.debug`), not only their effect on positions.
4. **High-level behaviour is benchmarked once, not kept as tests:** measure when a
   change lands, write the number here, move on.

### 9.2 Integration (next)

The first planet runs of the assembled model. Calibrate together, and record the
result once:
- **Targets:** plate speeds (rms ~4 cm/yr, slab-attached plates fastest); interior
  rigidity (non-rigid strain rate ≲ 10⁻⁹/yr); ridge depth ∝ √age; ~7 km oceanic
  crust; orogen heights bounded by collapse; coverage exact; volume budget closed;
  damage localised into narrow boundaries.
- **Handles:** basal drag (η_a), the GPE force (on), W_D and τ_heal, the edge
  interaction parameters (`ductile_damage_weakening`, cohesion, the tangential ratio,
  the aspect cap), and subduction's (interface weakening, the buoyancy criterion,
  `slab_attach_factor`).
- **Measure before changing the solver,** at spacings of 220, 110 and 50 km (≈10k,
  40k, 240k particles): iterations per step needed to hold the residual stress below a
  few MPa; plate rigidity under the mantle forcing (rms non-rigid speed of plate
  interiors against the ~0.1 mm/yr that elastic plates at MPa stresses imply; the
  last measurement, under older physics, gave ~1 mm/yr); convergence of the
  tangential rows at plate scale; whether caps slow Chebyshev in dense yielding zones;
  wall-clock cost per simulated Myr on a real GPU.
- **Decisions waiting on it:** the breakup criterion; whether to seed young ridges in
  the initial state (its oceans are all 60 Myr old, so nothing melts until stretched
  by β > 1.4); the aspect cap; the subduction buoyancy criterion.

Already measured on a generated planet with the assembled model (short runs): cells
sum to 4π − 0.08%, the deficit in ~2% clipped cells at voids of the initial state
wider than the hash reach (their neighbours take the void's area and thin, as
intended); no slab forms in 20 steps (then under the stress timer, since replaced by
the trench hinge); a 10× time step (1.6 Myr
per step) stays bounded (p99 speed 5.9 cm/yr, max 24); a record resumed in a fresh
page continues the run.

#### The two-plate test planet

Integration starts on a planet built so that every mechanism should happen once, in
order, at the coarsest resolution that still resolves them (`src/init/two-plate.js`,
the page's default world). Columns per feature, not particles per planet, set that
resolution: a continent needs ≥ 7 columns (one interior column), an ocean ≥ 4 from
ridge to trench (an age gradient), a ridge or trench ≥ 5 along its length, a slab
chain 2–3. That gives 100–150 columns on Earth (spacing ~2,000 km); the 162 vertices of
a twice-subdivided icosahedron are used, bonded along its edges with apothems fitted
to the bond lengths (largest initial strain 1e-5, coverage 99.4%; the 12 pentagons
leave the gaps).

- **Plates.** The simplest plate system on a sphere: two hemispherical plates turning
  relative to each other about an Euler pole P on their boundary. Half the boundary
  diverges (a ridge), half converges (a trench); the relative speed goes as
  sin(angle from P), so both taper to zero at P with no transform faults. Bonds across
  the boundary are broken.
- **Mantle** (`src/mantle/kinematic.js`): prescribed, not convecting. Plate B's mantle
  is at rest (the reference frame); A's turns about P at ω (8 cm/yr at 90° from P),
  with a tanh step of 0.05 rad at the boundary, so it upwells along the ridge half and
  downwells along the trench half and nowhere else.
- **Crust.** Seafloor age is the steady state of symmetric spreading, 2φ/ω, φ the
  angle around P from the ridge. B carries a 17-column continent along the trench half
  (an Andean margin); A carries a 7-column microcontinent placed to reach the trench
  after ~100 Myr at the mantle's rate. A's leading row on the trench half (9 columns)
  starts as slabs, in place: a slab is a column that sinks where it subducted (its
  edges need overlapping depth intervals), so it is seeded at the trench, not under
  the overriding plate.
- **Scaled parameters.** Lengths on the unit sphere scale with the column size
  (apothem 0.149 against 0.0175): `base_apothem`, `cell_clip_radius` (0.34),
  `arc_reach` (0.42), and 4 hash cells per axis (reach of the tessellation).

Measured (software GPU, 162 columns):
- **The solve did not converge at 32 iterations.** Summing every force on each
  surface column (edges + basal drag, `bal` diagnostic; the net-force arrow draws
  the same sum): at dt = 0.25 Myr the median net edge force was 11.5× the median
  drag, and plate interiors carried 40–60 MPa of compression that nothing balanced.
  Above the 50 MPa transition stress, with all old seafloor negatively buoyant, that
  set off subduction inside both plates, and each new slab shortened its
  neighbours' window to 0.1 Myr: a cascade (162 → 129 columns in 40 Myr at
  dt = 0.5; 162 → 71 at dt = 4). Residual / drag against iterations: dt 0.25: 32 →
  11.5, 128 → 1.3, 512 → 0.45 (worst compression −49, −12, −6 MPa); dt 1: 128 →
  9.0, 512 → 2.1; dt 4: 1024 → 4.9. At equal cost (iterations per Myr), small steps
  with more iterations win. Probably the same cause as the scattered subduction
  seen on the 10k planet; not yet measured there.
- **Time step:** 0.25 Myr; the solve, 256 iterations until the warm start (5.2), now
  the default 32 (8× cheaper, velocities within ~6% of a converged solve). Besides convergence, the slab's
  lifetime bounds it: slabs sink at w ∝ column width (`slab_sink_rate_km_myr`), so
  1,900 km wide slabs sink ~50 cm/yr and are gone in ~1.3 Myr, and their pull needs
  several steps.
- **With 32 cold-started iterations at dt ≤ 0.25 the trench locked** (with the warm
  start it is still locked over the first 16 Myr: both halves turn at 0.31–0.37°/Myr
  about the mantle's pole, i.e. ~3 cm/yr rms, until the net rotation is removed, 5.3,
  and then barely move): the seeded slabs drop out
  within 1.5 Myr, nothing else subducts, and A pushes B through the trench, both
  turning at ~3.3 cm/yr against A's mantle at 7 (B's mantle at rest).

#### The trench hinge: findings

A heuristic for the subduction rules, to hold as an outcome, not to be coded: *a
slab does not disappear before the tile it is dragging has started to subduct.*
Under the stress timer every seeded slab broke it: slabs sank at ~50 cm/yr (Stokes
size = map width), their edges went at ~100 km depth, and solver residual tripped
the timer. The trench hinge (section 6) replaces the timer; the heuristic is now a
counter (`slabsLost`). Measured:
- **Strip (default resolution, 220 km columns):** the heuristic holds. The leading
  column bends, goes 118 km under a continent and subducts at 1.5 Myr with its slab
  still hanging (GPU test, damage off; the strip moves ~30–50 cm/yr under ~10¹⁸ N of
  slab pull against the drag of eight columns).
- **Found on the way, and fixed:** (1) the edge law's brittle tension cap was
  cohesion alone, so a plate could transmit at most ~2×10¹² N/m, less than any
  lubricated trench contact resists: now Anderson's extension strength (4.5). (2) A
  contact that gave way only when the flap tip passed the overriding plate's base
  (100–150 km) could never start: now it gives way with dip, toward the optimal thrust
  angle. (3) Viscous support from the hang point's motion, and the lagged load, made
  load and bend flip every step: support now from the plate's feed rate, load
  under-relaxed. (4) Slabs left at their mantle position fell behind the advancing
  plate, and the "far-side" arc reset the subducting plate's lithosphere, which ended
  the edge behind the hinge: slabs now hang at the trench. (5) The derived Chebyshev ρ
  diverged under concentrated pull (columns jumped ~1,000 km in a step): positions
  extrapolated apart from clamped multipliers (fixed, 5.2).
- **Still open: the damage law tears plates under slab pull.** Shallow brittle slices
  crack under a few MPa of tension; their plastic work lowers the whole edge's
  integrity, the caps fall with it, and the edge behind the hinge fails (strip:
  integrity 1 → 0.14 in 0.4 Myr). A single integrity per edge couples shallow
  normal faulting to whole-plate strength. Candidates: damage from the load-bearing
  slices only, or integrity per slice. Decide with W_D (the handles above).
- **Two-plate planet (162 columns, 40 Myr):** all 9 seeded slabs hang at first, the
  hinges bend 4–23° and yield plastically; the slabs (1,600 km long, ~5× heavier than
  the plate can hold) neck, and five trench columns subduct within 8 Myr, but as free
  slabs: the damage law had already broken the edges behind them. Then the trench goes
  quiet; one slab counted lost while hanging. A 1,900 km column must also go ~1,800 km
  under before its hinge passes its back face, so subduction there is granular:
  whole-tile conversion is too coarse at this resolution. A resolution-free
  alternative: consume the column through its hinge, moving area and volume into its
  hanging slab as it goes under (the column shrinks from the front, its centroid moving
  back), so the slab is fed continuously. Not built.

### 9.3 Physics extensions

- **Faster bending solve.** Damped Jacobi needs ~(α/ℓ)⁴ sweeps. Chebyshev
  acceleration of the bending sweeps (the operator is symmetric positive definite; it
  needs its own spectral-radius bound) brings that to ~(α/ℓ)², enough at 50 km
  (α/ℓ ≈ 2–4: tens of sweeps); multigrid keeps the count roughly constant at finer
  spacing.
- **Plate-scale convergence of the in-plane solve,** if 9.2 shows the iteration count
  too costly: (1) a primal (position-based) formulation such as vertex block descent
  (Chen et al. 2024), where a plate-rigid initial guess and coarse corrections are
  legitimate; (2) hierarchical constraints (Müller 2008), coarser particles linked to
  fine ones; (3) keep Chebyshev and spend iterations.
- **Trench and outer rise:** couple the slab's vertical load to the plate behind it,
  as a line load in the bending solve.
- **Slab geometry:** a slab is one column hanging off a 3D-measured edge and detaches
  ~100 km down. A chain of slab columns, bent by the bending energy, would carry pull
  deeper and let rollback and trench retreat emerge.
- **Megathrust friction:** a tangential channel on the slab interface with its own
  (low) friction.
- **Plumes and hotspots:** an active-upwelling melt term, w_mantle, from mantle
  plumes, with a hotter, deeper melting depth: dV = A·w·F̄·max(1 − L/H_m, 0) at fixed
  area, giving oceanic plateaus and hotspot chains. Volcanism is a volume source; a
  ridge is an area source.
- **Crust → mantle feedback:** continents blanket (warm) the mantle beneath them;
  slabs become the mantle's cold downwellings, replacing random drips. The mantle
  generator then becomes part of L1 (it reads L1 state); a frozen recorded mantle
  remains available for reproducible benchmarks.
- **Toroidal mantle flow:** a vertical-vorticity part would drive spins as well as
  resist them.
- **Lateral heat conduction** between columns, and **sea level** responding to ridge
  volume.
- **Power-law creep** at the edge's actual strain rate instead of viscosity
  linearised at 10⁻¹⁵/s.
- **Thermal state of slabs** beyond the thermal-age decay; slab breakoff as an event.
- **Anisotropic weakening (fabric):** reduced cohesion on faces parallel to a column's
  long axis (foliation-parallel weakness), so old sutures and shear zones reactivate
  (the Wilson cycle). One parameter, off by default.
- **Strain-rate and stress tensors as measurements:** the deformation gradient fitted
  from neighbours (Fᵢ = (Σⱼ wᵢⱼ dᵢⱼ Dᵢⱼᵀ)(Σⱼ wᵢⱼ Dᵢⱼ Dᵢⱼᵀ)⁻¹, d current and D rest
  vectors), its polar decomposition Fᵢ = RᵢSᵢ, and the thickness-integrated stress
  tensor from the edges, for display, export and the split of edge stress into
  isotropic and deviatoric parts. The continuum rotation Rᵢ against the spin is a
  diagnostic of rolling in fault gouge.
- **Accumulated plastic strain** per column (diagnostic; inherited weakness for L2).
- **Adaptivity** (refine first, coarsen second). Refine (split, with the same
  machinery, volume-preserving and keeping the parent's material) where strain rate,
  proximity to a slab or damaged edge, or curvature exceeds a threshold; coarsen by
  merging two bonded columns of the same level in calm interiors, edges rewired
  stress-neutrally (the fiddly part). Neighbours differ by at most one level (area
  ratio ≤ 2, as in quadtree balancing); a per-particle `level`; the hash grid sized for
  the coarsest contact, or one grid per level. The implicit solve has no stability
  limit from small particles; prefer more iterations to substepping. Acceptance: the
  integration benchmarks give the same continuum answer at 220/110/50 km within stated
  tolerances.

### 9.4 Patch mode and the stack of models

Run L1 code on a projected regional patch driven by a forcing, with adapters from
records and an event log. Acceptance: a patch driven by a global record stays within
tolerance of the global fields in its interior over [t₀, t₁].

**Patch frame:** a projection centred on (lat₀, lon₀) (default azimuthal
equidistant), local units in km, local float32 coordinates; a child's forcing covers
[t₀, t₁] in the parent's time and the child runs its own clock and dt; depths
positive down from sea level, sea level its own time series.

**What L1 needs from above** (basal forcing): mantle velocity at the lithosphere base
v_m(x, t), basal heat flux or mantle temperature, optionally down-going regions.
Contract for swapping generators: `sample(x, t) → {v_m, q_basal}`, optionally reading
L1 state; a recorded mantle model (true L0) supplies the same fields with no feedback.

**What L2 (regional patch) needs from L1:**
- domain: centre, projection, extent, time window;
- initial state at t₀: crust and lithosphere thickness, composition, thermal age,
  plastic strain, plate ids and boundary context;
- lateral boundary forcing along ∂Ω over [t₀, t₁]: velocity and/or integrated stress,
  plus the material state of anything flowing in;
- basal forcing over Ω: mantle velocity, heat flux, slab presence and depth;
- globals: sea level(t); optional nudging targets for the relaxation zone.

**What L1 should guarantee to children:** fields continuous in time between samples
(linear interpolation valid); velocities and strain rates consistent (strain rate the
symmetric gradient of velocity on the record mesh); crust volume conserved by the
remap.

**Record fields to add:** lithosphere thickness or thermal age; strain-rate tensor
(1/Myr, 3 components); integrated stress tensor (MPa·km, 3 components); boundary type
and distance to the nearest boundary; subduction polarity and slab depth beneath;
basal heat flux; plastic strain / damage; sea level series; crust thickness remapped
conservatively (today it is kernel-averaged); optional equirectangular rasters.

**Events to add:** `rift_onset`, `breakup` (counted today, not logged),
`subduction_initiation`, `slab_breakoff`, `collision_onset`, `suture`, with time,
location (lat/lon or a polyline), plate ids and a magnitude.

**Tracers** (for history): Lagrangian markers recording provenance (birth time and
setting, peak pressure and temperature). The schema reserves a place.

**Generators** (procedural forcings in the record schema, doubling as validation
benchmarks; several already exist as mechanism tests):

| Generator | Purpose | Analytic expectation |
|---|---|---|
| rigid rotation about a pole | null test | zero strain everywhere |
| steady ridge at rate V | spreading, depth–age | depth ∝ √(distance / V) |
| uniform convergence | collision, thickening | thickness grows at rate (V/L)·h |
| line load on a plate | flexure | w(x) = w₀ e^(−x/α)(cos x/α + sin x/α) |
| thickened block | isostatic relaxation | settles to Airy height; time scale γ/(Δρ g A) |
| isolated plate + slab | slab pull | terminal speed = slab force / Σγ |
| noise-driven basal flow | game "world seeds" without a mantle model | – |

---

## 10. Open questions

- **Localisation:** does work-driven damage with thermal healing concentrate into
  narrow plate boundaries, or weaken plates diffusely? Bercovici's group finds
  localisation only in a window of healing rates. Decide W_D and τ_heal at
  integration.
- **Fault-zone width:** friction plus face rolling should localise slip onto lines of
  edges; whether that happens at 220 km spacing, and how wide the zones are.
- **Ductile tension at zero integrity:** a broken boundary between old plates resists
  opening at ~80% of ridge push through its ductile base (4.5). Raise
  `ductile_damage_weakening`? New hot edges are held by their surroundings and heal
  fast; columns made in a highly dynamic setting should not cohere (transform faults
  cut ridges).
- **Aspect cap:** is 3 right, and is neighbour exchange beyond it a good model of
  ductile shear zones at 50 km?
- **GPE force vs basal drag:** the force is physical but, against today's drag,
  metres per year at margins. Calibrate drag, edge tension and this force together.
- **Caps inside Chebyshev:** clamping λ to caps is a nonlinearity. With the
  positions following the multipliers it no longer parts them, but overestimating ρ
  still derails clamped rows; the 0.999 ceiling is one measurement (5.2).
- **Tangential rows outside Chebyshev:** plain Jacobi converges them in a few sweeps
  on a patch; does that hold at plate scale, where spins couple through the positions
  over many columns?
- **Subduction calibration:** freeboard < 0 makes ocean older than ~8 Myr subductable
  (intent: 30–40 Myr); interface weakening and the attachment strength are guesses.
  Set them with plate speeds.
- **Chebyshev ρ, settled (5.2):** derived, capped by the iteration budget and at
  0.999, with a warm start. Open: the ceiling is one strip; a solve for the slow modes
  (a coarse level, 9.3) would make the budget cap unnecessary.
- **Breakup criterion:** volume-mixed composition crosses into oceanic only at
  β ≈ 10 (a continent stretched 1% per step starts melting at β = 2.15 and crosses at
  10.6, with 6.6 km of crust); real margins break at β ≈ 3–5. Key breakup on the melt's
  share of the upper crust, or on lithosphere thickness?
- **Large strain per step:** the shape and area updates are first order in the step's
  strain: within a few percent at 5–7% per step (hand-driven tests); real steps are
  ~10⁻⁴.
- **Slab fate:** a hard removal depth is simple. Does trench behaviour depend visibly
  on the cutoff?
- **Plate labelling stability:** how to avoid id flicker for plates joined by a few
  weak edges.
- **Record mesh:** icosphere (uniform, awkward for tools) vs. HEALPix (Hierarchical
  Equal Area isoLatitude Pixelization: uniform and tool-friendly).
- **History:** are tracers enough, or do columns carry their own stratigraphy?

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Myr | million years |
| PBD / XPBD | position-based dynamics / extended PBD (Müller et al. 2007; Macklin, Müller & Chentanez 2016) |
| Chebyshev (semi-iterative) acceleration | extrapolating each iterate from the previous two with weights from Chebyshev polynomials, given a bound on the plain iteration's spectral radius |
| GPE | gravitational potential energy (per unit area of a lithospheric column) |
| LAB | lithosphere–asthenosphere boundary: the base of the plate |
| Euler pole | axis about which a rigid plate rotates on the sphere (Euler's rotation theorem) |
| Airy isostasy | each column floats independently at the height its density allows |
| flexural isostasy | columns coupled by bending stiffness, so loads spread over the flexural length α = (4D/Δρg)^¼ |
| Winkler foundation | restoring force proportional to vertical displacement, i.e. buoyancy |
| Haskell rebound time | the time a load of wavelength λ on a viscous half-space of viscosity η takes to relax: 4πη/(Δρ g λ) |
| Cosserat medium | a continuum whose points carry orientation as well as position |
| spin (of a column) | its rotation about the vertical, solved for with its position; the micro-rotation of the Cosserat medium |
| polar decomposition | writing a deformation gradient as rotation × stretch |
| finite-strain ellipse | the shape a small circle of material takes after deformation; its aspect ratio measures accumulated strain |
| logarithmic strain | ln(stretch); increments add for deformation along fixed axes |
| kern rule | masonry: a joint stays closed while the resultant force passes through its middle third, i.e. moment < force × length / 6 |
| parallel-axis theorem | moment of inertia (here, of drag) about an axis = about the centre + area × distance² |
| CF conventions | Climate and Forecast metadata conventions for gridded scientific data |
| HEALPix | Hierarchical Equal Area isoLatitude Pixelization of the sphere |
| Maxwell element | spring and dashpot in series: elastic on short times, viscous on times longer than η/E |
| yield-strength envelope | strength of the lithosphere versus depth: the weaker of brittle friction (rising with depth) and ductile creep (falling with depth as it gets hotter) |
| Byerlee friction | the empirical rock friction law, shear strength ≈ 0.6 × normal stress |
| dislocation creep | ductile flow of rock by crystal defects moving; strain rate ∝ stressⁿ exp(−Q/RT), n ≈ 3–5 |
| brittle–ductile transition | depth where creep becomes weaker than friction; follows an isotherm (~300 °C in quartz, ~650 °C in olivine) |
| jelly sandwich | continental strength profile: strong brittle upper crust, weak ductile lower crust, strong upper mantle |
| half-space cooling | conductive cooling of sea floor moving away from a ridge: T follows erf(z / 2√(κ·age)), so depth and lithosphere thickness grow as √age |
| ridge push | horizontal force from the GPE difference between young (hot, high) and old (cold, low) ocean lithosphere |
| tessellation cell (Voronoi cell) | the region of the sphere closer to a marker than to any other |
| power diagram | a Voronoi diagram with weights: each site's boundary moves toward the smaller neighbour. Here: planes x·(aⱼpⱼ − aᵢpᵢ) = 0, a = 1/cos(apothem) |
| gnomonic projection | projection of the sphere from its centre onto a tangent plane; great circles become straight lines |
| Sutherland–Hodgman | clipping a polygon by one half-plane at a time |
| Van Oosterom–Strackee formula | the solid angle of a triangle of three vectors: tan(Ω/2) = det / (products of lengths + dot products) |
| stretching factor β | ratio of stretched to original length (McKenzie 1978); here of area, A/A₀, which thins the lithosphere by 1/β |
| decompression melting | rising mantle melts as pressure drops below the solidus; ~10% melt from ~70 km gives 7 km of oceanic crust |
| freeboard | height a column would float above the mantle datum without water loading, ∫(1 − ρ/ρ_m)dz; negative = the column is denser than the mantle and can sink |
| flexural length α | (4D/Δρg)^¼: the distance over which a plate on a fluid foundation spreads a load; under a line load the plate first crosses zero at 3πα/4 and bulges up by e^(−π) ≈ 4% at πα |
| finite-volume ("cotangent") Laplacian | (∇²f)ᵢ ≈ (1/Aᵢ) Σⱼ (shared boundary/distance)(fⱼ − fᵢ) on a Voronoi mesh; for a Delaunay triangulation the weights equal the cotangent formula |
| Stokes sinking | settling of a dense body through a viscous fluid at a speed ∝ buoyancy × size / viscosity |
| suture | the weld where two colliding plates become one |
| magmatic arc | chain of volcanoes above a subducting slab at ~100 km depth, fed by melt from the mantle wedge |
| solidus | temperature (at a given pressure) at which rock begins to melt |
| T1 event | neighbour exchange in a foam or cell network: one contact is lost and another forms |
| integrity φ | an edge's undamaged fraction: scales cohesion |
| thin viscous sheet | lithosphere modelled as a depth-averaged viscous layer (England & McKenzie 1982); Argand number = buoyancy stress / viscous stress |
| downscaling, relaxation zone | regional climate modelling practice for driving a fine model with a coarse one at its boundaries |
