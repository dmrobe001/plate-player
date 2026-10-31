# Plate Player — design plan

Status: **phases 0–2 implemented**; the rest is a proposal. This document says where
the model is going and why.

**How to use this document.** It is a working plan, not a specification. Don't
pursue its goals too doggedly: when a phase runs into difficult problems (a method
that doesn't converge, a cost that doesn't fit, a requirement that turns out to fight
another), stop and discuss a redesign rather than forcing the plan through. Record
what changed and why in the phase table (section 9).

Contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Principles](#2-principles)
3. [State](#3-state-what-the-gpu-holds)
4. [Physics: one energy, one dissipation](#4-physics-one-energy-one-dissipation)
5. [The time step](#5-the-time-step)
6. [Boundaries, subduction and spreading](#6-boundaries-subduction-and-spreading)
7. [Resolution and adaptivity](#7-resolution-and-adaptivity)
8. [Contracts between scales](#8-contracts-between-scales)
9. [Implementation phases and acceptance tests](#9-implementation-phases-and-acceptance-tests)
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
  degradation at coarser resolution (~220 km, ~10k particles) for fast iteration.
  "Graceful" means it still runs and still looks tectonic. Statistics like trench
  widths may deviate from Earth's.
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
  per-particle fields and an event log (section 8.5).

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

3. **Boundaries live on edges.** Bonds and contacts are one structure, the *edge*.
   Plate-boundary character (ridge, trench, transform, suture) is either stored edge
   state or derived from edge kinematics.

4. **Two levels of resolution in the solver.** Plates are the coarse level: rigid
   rotations about Euler poles, solved first. Particles are the fine level and carry
   deformation within plates. This is aggregation multigrid with geologically
   meaningful aggregates.

5. **Material parameters are continuum quantities.** Young's modulus, Poisson's
   ratio, elastic thickness, viscosities and densities are given in physical units
   and converted to per-particle and per-edge constants using the local particle
   size. The material must not change when resolution changes.

6. **Levels talk only through records.** A model never reads another model's
   internals; it reads a *forcing* built from a parent's *record* or from a
   procedural generator, under a stated contract (section 8).

7. **Seeded randomness only**, as now.

---

## 3. State: what the GPU holds

Sizes are for the 50 km target: ~240k particles, ~1M edges, ≤256 plates.
Memory stays within a few hundred MB of storage buffers.

### 3.1 Particle = lithospheric column

A particle is a hexagonal prism of lithosphere. Its *position* is the centre of
its top surface and its *director* is its axis, which is radial for flat-lying
crust and tilted for slabs.

| Field | Type | Units | Notes |
|---|---|---|---|
| `position` | vec3 | unit sphere | **free radius** (new): \|x\| = 1 + surface height / R |
| `director` | vec3 | unit vector | column axis; carries bending (new) |
| `orientation` | f32 | rad | rotation about the director, from the polar decomposition (new) |
| `area` | f32 | unit sphere² | hexagon area; apothem = √(area / 2√3) |
| `crust_thickness` | f32 | km | |
| `lith_mantle_thickness` | f32 | km | cold mantle layer that moves with the crust (new) |
| `crust_density` | f32 | kg/m³ | at mantle temperature, as now |
| `age` | f32 | Myr | time since formation; drives oceanic cooling |
| `thermal_age` | f32 | Myr | reset by reheating (rifting, arcs); sets lith_mantle_thickness (new) |
| `plastic_strain` | f32 | – | accumulated irreversible strain; weakens the column (new) |
| `plate` | u32 | – | plate slot, from labelling (new) |
| `kind` | u32 | – | surface / slab / ghost (boundary-driven) / inactive |
| `level` | u32 | – | resolution level for adaptivity (new; 0 = base) |
| `strain` | 3×f32 | – | symmetric 2×2 strain from the neighbour fit (diagnostic + yield) |
| `stress` | 3×f32 | MPa·km | thickness-integrated stress tensor (diagnostic + export) |
| `x_prev`, `x_pred` | vec3 | | solver scratch |

Temperature is no longer a per-particle prognostic scalar. The column's thermal
structure follows from `thermal_age` (half-space cooling) plus a reheating rule for
slabs and arcs. That removes an explicit ODE and gives the depth–age law directly.

### 3.2 Edge = bond or contact

| Field | Type | Notes |
|---|---|---|
| `a`, `b` | u32 | particle indices |
| `type` | u32 | `BOND` (cohesive), `CONTACT` (touching, no cohesion), `SUBDUCTION` (contact with polarity) |
| `rest_a`, `rest_b` | 2×vec2 | rest edge vector in each particle's own frame; used by the strain fit and for bending |
| `damage` | f32 | 0..1, weakens cohesion; a BOND at damage 1 becomes a CONTACT |
| `polarity` | u32 | SUBDUCTION only: which end goes down |
| `lambda` | f32 | XPBD multiplier for this step |
| `friction_state` | f32 | Coulomb friction scratch (stick/slip) |
| `born` | f32 | Myr; for events and history |
| `stress_n`, `stress_t` | f32 | resolved normal and shear stress from the particle tensors (display) |

Edges are created in two ways. **Bonds** come from initial conditions, from splits,
and from welding (a contact held long enough under compression can become a bond:
suturing). **Contacts** are created by the collision pass when two non-bonded
columns overlap, and destroyed when they separate.

Boundary *classification* for display and export (divergent, convergent, transform)
is derived from the relative velocity across the edge, not stored.

### 3.3 Plate (coarse level)

| Field | Type | Notes |
|---|---|---|
| `omega` | vec3 | angular velocity about the Euler pole, rad/Myr |
| `torque` | 3×i32 atomic | fixed-point accumulation |
| `drag_tensor` | 6×i32 atomic | I_γ = Σ γᵢ (\|rᵢ\|² 𝟙 − rᵢ rᵢᵀ), symmetric |
| `area`, `centroid` | | for export |
| `persistent_id` | u32 | stable across re-labelling (section 5.3) |

### 3.4 Mantle

The mantle is a **basal-forcing generator** (section 8.4) on the CPU: the "lava lamp"
model in `src/mantle/`, described with its grounding in section 9, "Mantle and basal
drag". It supplies v_mantle(x, t) and its divergence; crust → mantle feedback
(blanketing, slabs as cold sinks) is not yet coupled.

---

## 4. Physics: one energy, one dissipation

Notation: A = column area, hₑ = elastic thickness (from `thermal_age` and crust
type), E = Young's modulus, ν = Poisson's ratio, g = gravity, ρ_m = mantle density.

### 4.1 Dissipation (the "mass" of the solver)

- **Basal drag:** γᵢ = Aᵢ η_a / h_a, with η_a and h_a the asthenosphere viscosity
  and thickness. It resists motion relative to the mantle velocity at the column.
- **Slab drag:** slab particles feel isotropic drag in 3D,
  γ_slab ~ (Stokes-like) η_m · (particle size).

### 4.2 In-plane elasticity (replaces the bond springs and the bogus shear)

For each particle i, fit the local deformation gradient by least squares over its
bonded neighbours j, comparing rest vectors Dᵢⱼ (stored in i's frame) with current
vectors dᵢⱼ (projected into i's tangent plane):

> Fᵢ = (Σⱼ wᵢⱼ dᵢⱼ Dᵢⱼᵀ)(Σⱼ wᵢⱼ Dᵢⱼ Dᵢⱼᵀ)⁻¹

Polar decomposition Fᵢ = Rᵢ Sᵢ. The rotation Rᵢ updates `orientation`, so rigid
rotation produces **zero strain**. The stretch gives εᵢ = Sᵢ − 𝟙. Energy, as
thickness-integrated plane stress:

> U_el = Σᵢ Aᵢ hₑ,ᵢ [ μ ε_dev:ε_dev + ½ K₂ (tr ε)² ],  μ = E / 2(1+ν),  K₂ = E / 2(1−ν)

This is a multi-particle XPBD constraint, involving i and its ≤6 neighbours. It
frees Poisson's ratio from the 1/3 that central springs on a triangular lattice
force, and gives a true stress tensor for yield, display and export.

### 4.3 Bending (new; needed at 50 km)

Directors nᵢ, with flexural rigidity D = E hₑ³ / 12(1−ν²):

> U_bend = Σ_edges (D Aᵢⱼ / 2ℓᵢⱼ²) |nⱼ − nᵢ − (rest difference)|²
>        + Σ_edges k_K (nᵢ · (xⱼ − xᵢ))²

The second term is a soft Kirchhoff constraint keeping the director normal to the
sheet. The discretisation is the least settled part of the design. It is accepted
only if it passes the line-load flexure benchmark (section 9) at 50 km.

### 4.4 Buoyancy (new; free radius)

Each surface column has an equilibrium surface radius r_eq,i from Airy isostasy of
its whole column (crust + lithospheric mantle, with temperature-dependent densities),
with water loading Δρ = ρ_m − ρ_w where submerged:

> U_iso = Σᵢ ½ Δρ g Aᵢ (rᵢ − r_eq,i)²

This is a Winkler foundation. With D = 0 every column sits at r_eq, recovering
today's Airy elevation. With D > 0, loads spread over the flexural length α:
foreland basins, forebulges and trench outer rises emerge.

Slab particles are below the lithosphere base and have no restoring equilibrium.
They feel a body force (ρ_col − ρ_m) g V along the local vertical, and sink until
assimilated.

### 4.5 Contact

Columns are prisms with radial extent [r_bottom, r_top]. Two columns are in contact
only if they overlap in map view **and** in depth. This is what lets a slab pass
beneath an overriding plate. Contacts are XPBD inequality constraints (non-penetration)
with Coulomb friction μ_f on the tangential part. The subduction interface uses a
low μ_f (it is weak on Earth).

### 4.6 Gravitational potential energy (ridge push, plateau collapse)

A thin-sheet column model loses the horizontal force from lateral differences in
gravitational potential energy (GPE). Add it as an explicit, smooth body force:

> Fᵢ = −Aᵢ ∇Γ,  Γ = ∫ (lithostatic pressure − reference column pressure) dz

The gradient comes from a least-squares fit over neighbours. With depth–age
cooling this produces ridge push, and on thick continents it produces plateau
collapse and orogenic spreading. Explicit is acceptable here because Γ varies
slowly; a test should confirm it has no stability limit in practice.

### 4.7 Irreversible processes (between solves)

- **Yield:** pressure-dependent (Byerlee-type, i.e. frictional) yield stress at
  shallow depth, Arrhenius-weakened ductile yield deeper. The depth-integrated yield
  envelope is evaluated per column from its thermal structure. Excess stress
  converts to `plastic_strain` (return mapping) and damages that particle's edges.
- **Damage healing:** exponential relaxation, faster when hot.
- **Volume conservation:** dilatation changes area; crust_thickness × area is
  conserved. Compression thickens (mountains) and extension thins (rifts), except
  where melt adds volume (section 6).
- **Thermal:** `thermal_age` advances; reheating rules reset it at arcs and rifts.

---

## 5. The time step

### 5.1 Schedule

```
1. thermal + irreversible     exponential updates; yield/damage return mapping
2. topology events            contacts from hash grid; edge state transitions;
                              subduction initiation; splits/merges; slab removal at depth
3. plate labelling            incremental min-label propagation over intact bonds
4. predict                    x̃ᵢ = xᵢ + v_mantle(xᵢ)·dt + F_ext,i·dt/γᵢ
5. coarse solve               per plate: I_γ Ω = Σ rᵢ × (γᵢ v_mantle,i + Fᵢ) ; move x̃ by rotation Ω·dt
6. fine solve                 k sweeps of XPBD over constraints (elastic, bending, buoyancy,
                              contact), graph-coloured Gauss–Seidel
7. finish                     v = (x − x_prev)/dt (diagnostic); orientation/director update;
                              stresses for display; recording if due
```

Steps 5 and 6 can alternate twice if benchmarks show the fine solve fighting the
coarse one.

### 5.2 XPBD with drag in place of mass

Multiplying Φ by dt gives ½|x − x̃|²_Γ + dt·U. For a constraint Cₖ with compliance αₖ
(energy Cₖ²/2αₖ):

> Δλₖ = (−Cₖ − α̃ₖ λₖ) / (Σᵢ wᵢ |∇ᵢCₖ|² + α̃ₖ),   Δxᵢ = wᵢ ∇ᵢCₖ Δλₖ
> with **wᵢ = 1/γᵢ** (instead of 1/mᵢ) and **α̃ₖ = αₖ/dt** (instead of αₖ/dt²)

Velocity is no longer state; it is a diagnostic. Large dt is safe; it only makes
each step more nearly quasi-static.

### 5.3 Plate labelling

- Every frame, run a few sweeps where each particle takes the minimum plate label
  over its intact BOND edges (damage < threshold). Labels persist between frames,
  so a few sweeps per frame keep up.
- Min-label propagation detects merges but **not splits**. Every K steps, re-seed
  labels from particle indices and propagate to convergence. Then map the new
  components to old `persistent_id`s by majority overlap: a small readback, done on
  the CPU. Split and merge events go to the event log.

### 5.4 Parallelism and determinism

- Graph-colour the edges (at most ~7 colours on a near-triangular lattice). Recolour
  on topology change, locally and greedily.
- Fixed-point integer atomics for accumulations (WGSL has no float atomics).
- Thread-timing-dependent behaviour (free-list pops, contact creation order) is
  listed in record manifests, not engineered away. Runs should agree statistically,
  not bit for bit.

---

## 6. Boundaries, subduction and spreading

**Spreading (kept, reinterpreted).** Where columns are pulled apart, area grows by
dilatation, the crust thins, and the gap fills with new 7 km oceanic crust from melt.
Concretely, the current grow-then-split rule stays, but:
- the parent keeps its *volume*, so its thickness drops;
- the child is new oceanic crust with `age` = `thermal_age` = 0.

Ridges stand high because young lithosphere is thin and hot (4.4), and ridge push
follows (4.6).

**Continental rifting → breakup.** Stretching continental crust thins until a
threshold, then behaves as above. The first oceanic child is logged as a breakup
event.

**Subduction initiation.** A convergent CONTACT edge becomes SUBDUCTION when:
1. one side's column is negatively buoyant relative to the mantle (old ocean,
   ≳30–40 Myr); and
2. normal stress across the contact exceeds a threshold, sustained over a time
   window (a stored accumulator).

Polarity: the denser column goes down. If both qualify, the older one goes down.
Continent never goes down while it remains buoyant; continent–continent convergence
produces collision (thickening) instead.

**Subduction (explicit slab).** The downgoing column's `kind` becomes slab. It
keeps its bonds to the trailing plate, so slab pull is transmitted through real
edges. It sinks under its own negative buoyancy, bends via the director energy, and
its depth interval no longer overlaps the overriding plate's, so they slide past
each other with interface friction. At a cutoff depth (e.g. 660 km), or once
thermally assimilated, the particle is removed and its edges retired. The slab is
the memory: polarity persistence and the slab-pull–speed feedback come from the
geometry, not from bookkeeping.

**Arcs.** Overriding columns within a set distance of a SUBDUCTION edge gain crust
volume at a rate proportional to the subduction flux, and their `thermal_age` is
reset (hot, weak arc lithosphere). This is how continents grow over time.

**Collision and suturing.** Buoyant columns in sustained compressive contact thicken
by volume conservation. A contact held long enough can weld into a BOND (suture),
merging plates (a logged event).

**Coarse-resolution fallback (~220 km).** Slabs are only 2–3 particles long, bending
is aliased, and arcs are one particle wide. The same rules still run; slab pull is
just lumpier. No separate code path. Deviation from Earth statistics is accepted.

---

## 7. Resolution and adaptivity

- **Base resolution** is set by `base_apothem`. All continuum parameters are scaled
  per particle and per edge from physical values (principle 5).
- **Refinement:** split a particle (same machinery as spreading, but
  volume-preserving and keeping the parent's material) where any of these exceeds a
  threshold:
  - strain-rate magnitude;
  - proximity to a SUBDUCTION or damaged edge;
  - curvature (|∇n|).
- **Coarsening:** merge two bonded same-plate particles of the same level in calm
  interiors. Their bonds rewire to the merged particle, with rest vectors recomputed
  from current geometry, so the merge is stress-neutral. Merging is the fiddly
  operation; implement it after refinement works.
- **Level ratio:** neighbouring particles differ by at most one level, so area
  ratio ≤ 2, as in quadtree balancing.
- **Hash grid:** cell size ≥ the largest contact distance, so the coarsest level sets
  it. If levels differ a lot, use one grid per level, or a grid sized for the finest
  level plus multi-cell insertion for coarse particles.
- **Time step:** the implicit solve has no stability limit from small particles.
  Accuracy near fine particles may want smaller dt; prefer more iterations to
  substepping.
- **Precision:** float32 on the unit sphere resolves ~0.4 m, enough for this level.
  Lower levels use local coordinates (section 8.2).

---

## 8. Contracts between scales

### 8.1 Vocabulary and rules

- **Level:** one model at one scale, e.g. L0 mantle (optional), **L1 global plates
  (this model)**, L2 regional crust patch, L3 local terrain…
- **Record:** what a run writes, self-describing and versioned.
- **Forcing:** what a run reads.
- **Adapter:** turns a parent's Record into a child's Forcing (cropping,
  reprojection, resampling). This is the only code that knows about two levels.
- **Generator:** produces a Forcing procedurally or analytically, with no parent run.
  A Generator and an Adapter must produce forcings with the same schema, so a child
  cannot tell them apart.

Rules:

1. **One-way.** Children never alter parents. Disagreement is measured and
   reported, never fed back.
2. **Fields, not particles.** Children consume Eulerian fields sampled on a stated
   mesh, so they are independent of the parent's discretisation. Lagrangian
   snapshots exist for restart and debugging; no child depends on them.
3. **Every quantity declares** its units, reference frame, sampling (point value vs
   cell average) and extensiveness. Extensive quantities (volume, mass) are remapped
   conservatively; intensive ones (velocity, temperature) are interpolated.
4. **Time resolution is part of the contract.** A record states its sampling
   interval. A child must not depend on parent variability faster than that; that is
   the child's own physics.
5. **Records are traceable.** The manifest names the code, seed and parameters, so the
   run can be regenerated to statistical (not bitwise) agreement (section 8.3).
6. **Policy for disagreement:** the parent is authoritative at the child's lateral
   boundary; the child is authoritative in its interior. Between them sits a
   relaxation zone, where the child is nudged toward the parent over a stated width
   and time scale. This mirrors the lateral boundary conditions of regional climate
   models ("dynamical downscaling", Davies relaxation), a field with decades of
   practice on exactly this problem.

### 8.2 Frames and time

- **Global frame:** unit sphere, body-fixed (rotation is not modelled at L1).
  Latitude/longitude derived from it, conventions stated in the manifest.
- **Patch frame:** a projection centred on (lat₀, lon₀) with a stated projection
  (default: azimuthal equidistant), local units in km, local float32 coordinates.
- **Time:** Myr since the start of the run that produced the record (`t_model`), plus
  an offset to a shared epoch if a stack wants one. A child's forcing covers
  [t₀, t₁] in the parent's time; the child may run its own clock and dt.
- **Vertical:** heights in km relative to sea level, with sea level as its own time
  series. Depths are positive down from sea level.

### 8.3 Record contents (what a level exports)

Four tiers; a run can be configured to write any subset.

**a. Manifest** (always): schema version, model name, code version (git commit),
seed, full parameter set, resolution, frames, time range, list of included tiers
and fields, and known nondeterminism.

**b. Field series** (the main product): fields sampled every Δt_rec (default 1 Myr at
L1) on a fixed mesh. Proposed mesh: an icosphere at least as fine as the particles,
plus optional derived equirectangular rasters for convenience.

| Field | Units | Kind | Needed by |
|---|---|---|---|
| surface height | km rel. sea level | intensive | everything |
| crust thickness | km | extensive (× area) | L2 initial state, inflow |
| lithosphere thickness / thermal age | km / Myr | intensive | L2 thermal state, strength |
| crust density / composition class | kg/m³, enum | intensive (mode for enums) | L2 material |
| surface velocity | km/Myr, 2 tangent comps | intensive | L2 boundary conditions |
| strain rate (tensor) | 1/Myr, 3 comps | intensive | L2 boundary conditions, nudging |
| integrated stress (tensor) | MPa·km, 3 comps | intensive | traction boundary conditions |
| plate id | persistent id | enum | context, game |
| boundary type and distance to nearest boundary | enum, km | intensive | L2 tectonic context |
| subduction polarity / slab depth beneath | enum, km | intensive | L2 arcs, basal context |
| basal heat flux | mW/m² | intensive | L2 thermal |
| mantle velocity at base | km/Myr | intensive | L2 basal forcing |
| plastic strain / damage | – | intensive | inherited weakness in L2 |
| sea level (scalar series) | km | global | everything |

**c. Snapshots** (sparse): the complete Lagrangian state (particles, edges, plates)
every Δt_snap. This is enough to restart a run exactly, and lets a child start from
an exact cropped state if it shares this model's code.

**d. Events** (cheap, the seed of later history): an append-only log of discrete
happenings, each with time, location (lat/lon or a polyline), involved plate ids and
a magnitude. Types: `rift_onset`, `breakup`, `subduction_initiation`,
`slab_breakoff`, `collision_onset`, `suture`, `plate_split`, `plate_merge`.

**e. Tracers** (later, for history): Lagrangian markers recording provenance
(birth time, birth setting, peak pressure/temperature). Deferred; listed so the
schema reserves a place.

**Storage:** keep it boring and portable. A directory (or zip, for browser
downloads) holding `manifest.json`, plus one little-endian binary array per field
per time block, with shapes and dtypes declared in the manifest. Field metadata
should follow climate-and-forecast (CF) naming conventions (`units`, `long_name`,
`standard_name` where one exists), so NetCDF or Zarr conversion is a renaming, not
a redesign.

### 8.4 Forcing contracts (what a level needs)

**L1 (this model) needs from above: a basal forcing.**
- Mantle velocity at the lithosphere base, v_m(x, t).
- Basal heat flux or mantle temperature.
- Optionally, down-going flow regions (where slabs are allowed to sink).

Today this comes from the internal SH generator, which also takes feedback from the
crust. That feedback means it is part of L1 rather than a separate level. Contract
for swapping: the generator implements `sample(x, t) → {v_m, q_basal}` and may
optionally read L1 state. A recorded mantle model (true L0) would supply the same
fields with no feedback.

**L2 (regional patch) needs from L1:**
- **Domain:** centre, projection, extent, and time window [t₀, t₁].
- **Initial state at t₀** on the patch: crust and lithosphere thickness, composition,
  thermal age, plastic strain, plate ids and boundary context.
- **Lateral boundary forcing** along ∂Ω over [t₀, t₁]: velocity (kinematic boundary
  condition) and/or integrated stress (traction boundary condition), plus the
  material state of anything flowing *in* (thickness, age, composition).
- **Basal forcing** over Ω: mantle velocity, heat flux, slab presence and depth.
- **Globals:** sea level(t).
- **Nudging targets** (optional): interior fields for the relaxation zone.

**What L1 should guarantee to children:**
- Fields are continuous in time between samples (linear interpolation is valid).
- Velocities and strain rates are consistent (strain rate is the symmetric gradient
  of velocity on the record mesh).
- Crust volume is conserved by the remap.

### 8.5 Generators (procedural and test forcings)

The same schema, produced from formulas. They double as validation benchmarks, since
many have analytic answers:

| Generator | Purpose | Analytic expectation |
|---|---|---|
| rigid rotation about a pole | null test | zero strain everywhere |
| steady ridge at rate V | spreading, depth–age | depth ∝ √(distance / V) |
| uniform convergence | collision, thickening | thickness grows at rate (V/L)·h |
| line load on a plate | flexure | w(x) = w₀ e^(−x/α)(cos x/α + sin x/α) |
| thickened block | isostatic relaxation | settles to Airy height; time scale γ/(Δρ g A) |
| isolated plate + slab | slab pull | terminal speed = slab force / Σγ |
| noise-driven basal flow | game "world seeds" without a mantle model | – |

### 8.6 Downward to the game

The game is just another consumer. Height maps, biome-relevant fields (crust age,
plate boundaries, volcanic arcs) and the event log are rendered from the same
records. Nothing game-specific lives in the simulation.

---

## 9. Implementation phases and acceptance tests

Each phase ends with a check that could fail. Two kinds:

- **Mechanism tests** are cheap (seconds to a minute), small, and about one
  mechanism with a known answer. They stay in `test/`.
- **Benchmarks** of high-level behaviour (dt sensitivity of a whole planet, plate
  speeds, flexure profiles at 50 km) are run once when the phase lands, and the
  number is written in the table below. They are *not* kept as regression tests:
  they are expensive and the behaviour they measure is meant to evolve.

| Phase | Work | Acceptance test |
|---|---|---|
| 0 ✅ | Housekeeping: exponential integrators for all rates; cache nearest mantle node; fix free-list and state-flip races; mantle substep from a stability bound, mantle time tied to Myr | Every slot free or active exactly once; cached node = nearest; no blow-up at 10× dt. dt sensitivity is *reported*, not asserted: PBD stiffness still depends on dt until phase 2, so that number is the baseline phase 2 must beat |
| 1 ✅ | **Minimal recording:** manifest + field series + snapshots (tiers a–c), download as zip, resume from zip | Restart from a record ZIP in a fresh page continues the run (bitwise at the time; now checked to a tolerance, since bit-identity is not a goal) |
| 2 ✅ (partly) | **Overdamped XPBD core:** drag as mass, real compliance. *Changed:* λ-consistent Jacobi instead of graph-coloured Gauss–Seidel (see below) | Two-particle relaxation matches implicit Euler, (1 + dt/τ)⁻ⁿ with τ = γ/2k, to 0.1% at dt/τ = 0.1, 1, 10 (kept as a test). Whole-planet dt-independence **not met**: blocked on plate-scale convergence (findings below) |
| 3 | **Unified edges + column contacts + free radius & buoyancy** | Thickened block relaxes to Airy height (Generator: thickened block) |
| 4 | **Neighbour-fit strain + orientation**, stress-tensor glyphs | Rigid rotation → strain < 10⁻⁶; uniaxial stretch → set Poisson ratio ±5% |
| 5 (identification ✅; coarse solve: redesign question) | **Plate coarse level** (labelling, Euler poles) | Labelling, persistent ids, split/merge events and Euler-pole fits are unit-tested on synthetic networks. The coarse *solve* does not fit the XPBD formulation (findings below) |
| 6 | **Lithosphere column:** thermal age, depth–age, GPE force | Ridge Generator reproduces depth ∝ √age |
| 7 | **Subduction:** initiation rule, slab particles, bending directors | Line-load flexure profile within 10% at 50 km; plate speed correlates positively with slab-attached perimeter in free runs |
| 8 | **Volume conservation, collision, suturing, arcs** | Uniform-convergence Generator thickens at the predicted rate; continent never subducts |
| 9 | **Adaptivity** (refine first, coarsen second) | Benchmarks 3–8 give the same continuum answer at 220/110/50 km within stated tolerances |
| 10 | **Patch mode:** run L1 code on a projected patch driven by a Forcing; adapters from records; event log | Patch driven by a global record stays within tolerance of the global fields in its interior over [t₀, t₁] |

The order is chosen so that phases 0–4 already improve the current game output at
220 km, before 50 km is needed.

### Phase 2 findings (measured once, 30 steps = 0.48 Myr, seed 1, SwiftShader)

- **Physical scales.** With E = 70 GPa, hₑ = 30 km, η_a = 10¹⁹ Pa·s, h_a = 100 km, a
  single bond relaxes in τ ≈ 40 yr against dt = 16 kyr: every step is quasi-static.
  Stress diffuses through a drag-coupled elastic sheet at D = E·hₑ·h_a/η_a ≈ 20 m²/s,
  i.e. ~3000 km (≈15 particle spacings) per step. So the *physical* plate is nearly
  rigid over one step; the *numerical* one is not, because each Jacobi sweep moves
  information one bond.
- **Bond–contact conflict was the largest stress source.** With the mantle switched
  off, the residual overlaps from the initial packing alone gave a median 31 MPa and
  p99 630 MPa after one step at 70 GPa. Fixed in the initial state (apothems shrunk to
  fit, median by 0.3%): with the mantle off, stress is now exactly zero.
- **Remaining stress is under-convergence.** Mantle on, 70 GPa: median |σ| 56 MPa, 13%
  of bonds over yield, damaged bonds 9.2% → 10.4% in 0.48 Myr (4 iterations). 16
  iterations: 38 MPa, 5.5%. Fourfold iterations buy only 1.5×, as diffusion-like
  propagation predicts. A converged solve at today's 1–2 cm/yr should give a few MPa.
- **Default E = 2 GPa as a stop-gap** (median 1.6 MPa, no spurious yielding) until the
  plate level converges stress.
- **Why Jacobi, not graph-coloured Gauss–Seidel:** colouring has to be maintained as
  splits and rewiring change topology, for a gain of ~2× in convergence rate. That
  doesn't change the (L/ℓ)² scaling that is the actual problem, so it isn't worth the
  bookkeeping yet. Multipliers are scaled by 1/(bonds at either end) and accumulated
  in full, so a converged Jacobi solve is the true energy minimum.
- **Whole-planet dt sensitivity** (rms position difference between dt = 0.16 and 0.016
  over 1.6 Myr, relative to rms displacement): 0.37, measured *before* the
  initial-overlap fix, against 0.30 in phase 0. Not re-measured; it can't approach 0
  until plate-scale stress converges.
- **Practical:** queuing many slow steps tripped Chromium's GPU watchdog (device lost);
  `runSteps` now keeps at most two steps in flight.

**Re-ordering (agreed):** Chebyshev acceleration of the Jacobi sweeps first, then
phase 5 (plates as a coarse level), then phases 3–4.

### Phase 5 findings: plate identification done, coarse solve reconsidered

- **Identification** (`src/plates/`): plates are connected components of intact bonds,
  found exactly by union-find on the CPU every 0.25 Myr (milliseconds for 10⁴
  particles; no extra GPU state). Ids persist by majority inheritance, which yields
  `plate_split`, `plate_merge` and `plate_lost` events (the event log of §8.3 tier d).
  Records gain a `plate_id` field (nearest-particle sampling; ids are not averaged), a
  plate time series and `events.json`.
- **Initial condition fix:** the Voronoi damage bands didn't actually separate plates.
  Bonds crossing a border with their midpoint just outside the damaged core kept
  damage < 0.5, so seed 1 was one global plate. Bonds joining different Voronoi cells
  now start fully damaged: 12 plates for every seed tried.
- **Kinematics** are fitted to each particle's displacement over the labelling interval
  (a finite rotation), not the one-step velocity, which at ~1 cm/yr is comparable to the
  solver's per-step tolerance.
- **The coarse rigid-body solve doesn't fit the drag-as-mass XPBD formulation.**
  Positions always satisfy x − x̃ = W Bᵀλ, and central bond forces exert no net torque,
  so every plate's drag-weighted net rotation is *already exact at every iteration*.
  What converges slowly is the distribution of force within a plate (the multipliers).
  Moving positions rigidly without a matching λ would change the physics, and a
  matching λ requires the global solve we are trying to accelerate. Chebyshev (32
  iterations, ~3 MPa residual) has taken the coarse level's convergence job at 220 km.
- **Measured (seed 1, early run):** plates move at ~0.5 mm/yr rms, because the mantle
  pattern hasn't developed. The non-rigid part is 0.1–0.3 mm/yr, about 50 m per
  0.25 Myr, consistent with elastic flexing at MPa stresses, and evenly spread over
  intact interiors and damaged edges (not floppy edges). Rigidity can't be judged
  until the mantle actually drives the plates.
- **Options for the coarse level**, if it is needed at 50 km where iterations grow
  as √(κ·dt) (~4× more):
  1. a primal (position-based) formulation, e.g. vertex block descent (Chen et al.
     2024), where a plate-rigid initial guess and coarse corrections are legitimate;
  2. hierarchical constraints (Müller 2008), coarser particles linked to fine ones;
  3. keep Chebyshev and spend iterations; measure at 50 km before deciding.

### Chebyshev acceleration (between phases 2 and 5)

Semi-iterative acceleration (Wang 2015): x_{k+1} = ω_k (x̂ − x_{k−1}) + x_{k−1}, where
x̂ is one plain sweep, applied with the same weights to the multipliers so that
x − x̃ stays consistent with λ. ω follows Wang's recurrence, starting after 3 plain
sweeps.

- **The spectral radius can be derived, not tuned.** The sweep acts on the
  multipliers as I − (BWBᵀ + α̃)/(d(w_a + w_b + α̃)), where B is the bond–particle
  incidence. Self-equilibrated bond-force patterns (null space of Bᵀ) are slowest:
  ρ = 1 − α̃/(d(2 + α̃)), with α̃ = 1/(κ·dt) and d = 6. That is 0.9996 at 70 GPa and
  dt = 16 kyr: plain Jacobi needs ~6000 sweeps per 10× error reduction, Chebyshev ~110.
  The chain test's empirical optimum (0.999 at d = 2) matches the formula (0.99875).
- **Mechanism test (kept):** a 61-particle pre-compressed chain at κ·dt = 200 against
  the exact tridiagonal solution. After 128 sweeps the error is 0.5% with Chebyshev and
  76% with plain Jacobi; plain Jacobi still has 14% after 1024 sweeps.
- **Two things broke acceleration on the planet**, both nonlinearities inside the
  iteration. Each made 64-iteration runs diverge (all bonds damaged, ~280 MPa):
  1. damage updated inside the iterations, so Chebyshev's intentional overshoot
     softened bonds; moved to a single pass after the solve (which design §4.7 wanted
     anyway);
  2. damaged bonds switching stiffness between tension and compression mid-solve; the
     switch is now decided by the previous step's stress, so the operator is linear
     within a step (a one-step lag when a damaged bond goes from compression to
     tension). Collisions are the same kind of switch and have not caused trouble so
     far; they may in strongly convergent zones.
- **Planet benchmark** (70 GPa, 20 steps, initial damage present): median |σ|, p99,
  share over yield. Plain Jacobi 16 iterations: 32 MPa, 185, 2.6%. Chebyshev 32: 3.0
  MPa, 19, 0%. Chebyshev 64: 1.3 MPa, 7, 0%. Without initial damage, Chebyshev 64 gives
  1.1 MPa, which is the expected converged level at 1–2 cm/yr. **E is back to 70 GPa**
  with 32 iterations as the default; the 2 GPa stop-gap is gone.
- **Cost:** iterations scale as √(κ·dt) per step, so the cost *per Myr* scales as
  1/√dt. Longer steps are cheaper per Myr, up to the point where per-step topology
  changes (splits, subduction) need finer time resolution.
- **What is left for the plate level:** the slowest modes are near-rigid motions of
  large patches (drag-dominated, little elastic energy), which a per-plate rigid-body
  solve removes directly; and the √(κ·dt) iteration count, which grows at 50 km
  spacing and at larger dt.

---

### Before changing the solver again: measure at higher resolution

Agreed: no further solver redesign (primal formulation, hierarchical constraints)
until the work needed at higher resolution has been measured. Measure, at particle
spacings of 220, 110 and 50 km (≈10k, 40k, 240k particles):

- iterations per step needed to hold the residual bond stress (median, p99) below a
  few MPa, with Chebyshev acceleration;
- plate rigidity under the lava-lamp mantle forcing: rms non-rigid speed of plate
  interiors, against the ~0.1 mm/yr scale that elastic plates at MPa stresses imply;
- wall-clock cost per simulated Myr on a real (not software) GPU.

Expectations to check: the bond-to-drag ratio κ scales as 1/spacing² (drag ∝ particle
area, bond stiffness independent of spacing), so κ·dt is ~19× larger at 50 km, and
Chebyshev iterations grow as √(κ·dt), about 4.4×. Particle count grows 22×. If the cost
per Myr is acceptable, keep the solver; if not, that measurement decides between
the options in "Phase 5 findings".

### Mantle and basal drag

Goal: ground each piece in something measurable, and say what is established
and what is a modelling choice.

**What the plates feel: Couette shear across the asthenosphere.** Each particle is
dragged by traction η·(v_mantle − v_plate)/h. Here h is the thickness of the
low-viscosity channel between the plate's base (the lithosphere–asthenosphere boundary,
LAB) and the depth where viscosity rises again. In the solver this is the per-particle
drag γ = A·η/h, so plate speed is *not* slaved to the mantle. It comes out of the force
balance between drag, the bonds holding the plate, collisions, and later slab pull and
ridge push.

| Ingredient | Value | Status |
|---|---|---|
| Asthenosphere viscosity η | 10¹⁹ Pa·s (range 10¹⁸–10²⁰) | established range; **the effective drag coefficient to tune** |
| Base of the low-viscosity layer | 300 km | established order (~200–400 km); modelling choice |
| Oceanic LAB depth | 2.32√(κ·age), κ = 10⁻⁶ m²/s, capped at 100 km | established (half-space cooling, plate-model cap) |
| Continental LAB depth | 150 km (cratons reach ~250) | established order; one value for now |
| Couette (linear shear) profile | | modelling choice; the qualitative consequence, that deep keels couple more strongly, is a common argument in the literature |
| Plate speeds today | 1–10 cm/yr, rms ~4 cm/yr; fast plates are those with slabs (Forsyth & Uyeda 1975) | established; the second point needs slab pull (phase 7) |

**The mantle: a "lava lamp".** The requirements were to be alive (evolving, never
locking into one flow map), to have realistic flow speeds where they matter (beneath
the plates), and to start from a settled state rather than an artificial one, without
solving mantle fluid dynamics.

- *Why Swift–Hohenberg went:* it is a gradient system (it minimises a Lyapunov
  functional), so it relaxes to a stationary pattern by construction. That is
  exactly the "locked-in" behaviour to avoid.
- *State:* a buoyancy field b (mid-mantle temperature anomaly) on a 2562-node
  icosphere (~480 km spacing).
- *Flow:* poloidal, v = s∇Φ with (∇² − 1/ℓ²)Φ = b: hot regions diverge (upwell), cold
  regions converge. Flow beneath plates being poloidal is physical: buoyancy drives
  poloidal flow, and Earth's toroidal surface motion comes from the plates' rigidity
  and boundaries (established). The screening length ℓ is the mantle depth (2900 km),
  because flow driven by an anomaly in a layer of depth H dies off over ~H (modelling
  choice). Without screening, the flow was dominated by the planet-wide imbalance of
  plume positions, and the rms speed wandered by factors of 3 between seeds.
- *Evolution:* semi-Lagrangian advection of b by its own flow (upwellings broaden,
  downwellings narrow, as in convection), a small effective diffusion, decay with a
  plume lifetime of 120 Myr, and Poisson births of hot plumes and cold drips (0.3/Myr
  each, so ~35 active, like Earth's ~40 hotspots; order of magnitude).
- *Speed calibration:* the scale s comes from the plume statistics, not tuning. The
  expected mean-square flow is Σ rate·(lifetime/2)·E/(4π), where E is one plume's flow
  energy (one screened solve per plume kind at start-up). Advection sharpens
  downwellings, so the realised speed is higher by a measured factor of 1.69 (6
  seeds × 400 Myr), which is divided out. Target: long-run rms 4 cm/yr. Realised: 5.8–7.6
  cm/yr per seed before the correction, so roughly ±20% around the target after it;
  the instantaneous rms fluctuates ~2× over 100 Myr.
- *Spin-up:* 400 Myr (over three plume lifetimes) alone before the crust starts, in
  ~2 s.
- *Alive:* the buoyancy pattern decorrelates in ~100 Myr (correlation 0.35 after 50
  Myr, ~0 after 100). Unit-tested as correlation < 0.5 after 250 Myr.
- *Not yet coupled:* crust → mantle feedback (continents warm the mantle beneath them;
  slabs should become the dominant cold downwellings, replacing random drips).

**Measured: driven plates, and what deforms them** (seed 1, 70 GPa, 8 warm-up + 16
measured steps; non-rigid = rms of v − ω×r per plate; interiors = 5+ bonds from a broken
bond; one-off, SwiftShader):

| configuration | median plate speed | non-rigid, all | non-rigid, interiors | damaged bonds (start 9.4%) |
|---|---|---|---|---|
| Chebyshev 32 iterations, growth on | 2.5 cm/yr | 1.20 cm/yr | 0.47 cm/yr | 11.1% |
| Chebyshev 32, **growth off** | 1.0 cm/yr | 0.38 cm/yr | 0.13 cm/yr | 9.4% (none new) |
| plain Jacobi 32, growth off | 2.4 cm/yr | 1.09 cm/yr | 0.94 cm/yr | 10.6% |
| plain Jacobi 64, growth on | 2.1 cm/yr | 1.02 cm/yr | 0.80 cm/yr | 10.6% |

- **The spreading proxy is the main source of deformation and damage.** It grows
  particles *inside* plates over upwellings; bonds keep their rest lengths while
  collisions push to the new apothem sum, injecting strain (~0.4% per 0.4 Myr, i.e.
  ~250 MPa at 70 GPa). That is also why more Chebyshev iterations looked *worse* in an
  earlier run (128 iterations: non-rigid 2.8 cm/yr, damage 19.6% after 48 steps): a
  better-converged solve resolves the conflict harder. Real crust is not created inside
  plates, only where plates separate.
- **Chebyshev does pay off:** without growth, interiors are ~7× more rigid than with
  plain Jacobi at the same iteration count.
- **Remaining non-rigidity** (~1 mm/yr in interiors, against ~0.1 mm/yr expected for
  elastic plates at MPa stresses) is under-convergence: the measurement to repeat at
  higher resolution (above).
- **Plate speeds** with the 4 cm/yr mantle: 1–6 cm/yr per plate (median ~1 cm/yr
  without the growth artefact). Earth's rms is ~4 cm/yr, but the fast plates are
  slab-driven, and slab pull is not in the model yet.
- **Decision needed:** how new crust is made. Options: (1) restrict growth to particles
  at opening plate boundaries (next to broken bonds under tension), with bonded pairs
  exempt from collisions (part of phase 3); (2) proper gap filling at divergent
  boundaries (phase 8: new particles placed in gaps, no inflation at all).
- **Change made first (user's call):** a bond's rest length is now always the sum of
  its apothems, whatever changes a particle's area, so the conflict above cannot arise.
  The initial state fits positions and apothems jointly to keep the start stress-free.
  Hypothesis to test: growth no longer tears plate interiors. Predicted: with growth on,
  interior non-rigid speed and new damage approach the growth-off row. Growth inside
  plates still *pushes* (an inflating particle now pushes through its bonds as well as
  its contacts), so some deformation is expected; the question is whether it breaks
  bonds. To be measured.

**Spreading proxy, regrounded.** Crust over diverging mantle grows in area at the
divergence rate (above 0.002/Myr, capped at 5%/Myr). Upwelling divergence is ~1%/Myr,
so global crust production is ~0.5%/Myr of the surface, against Earth's ~0.6%/Myr
(~3 km²/yr). This is still a proxy until phase 8 ties creation to where plates actually
separate.

## 10. Open questions

- **Bending discretisation** (4.3): directors with a Kirchhoff penalty vs. dihedral
  constraints on edge pairs. Decide by the flexure benchmark.
- **Coarse/fine coupling:** is one coarse solve per step enough, or do they need to
  alternate? Decide by the phase-5 benchmark.
- **Contact forces in the coarse solve:** inter-plate forces lagged by one step
  (explicit) is simplest. Check it does not oscillate at convergent boundaries.
- **Plate labelling stability:** how often to re-seed; how to avoid id flicker for
  plates joined by a few weak bonds.
- **Slab fate:** a hard removal depth is simple. Does it make trench behaviour
  depend visibly on the cutoff?
- **Mantle generator:** keep SH as L1-internal with feedback, or freeze it to a
  recorded field for reproducible benchmark runs? Likely both, by configuration.
- **Record mesh:** icosphere (uniform, awkward for tools) vs. HEALPix (Hierarchical
  Equal Area isoLatitude Pixelization: uniform and tool-friendly) for field series.
- **History:** when it returns, are tracers enough, or do columns carry their own
  stratigraphy?

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Myr | million years |
| PBD / XPBD | position-based dynamics / extended PBD (Müller et al. 2007; Macklin, Müller & Chentanez 2016) |
| SH | Swift–Hohenberg: a pattern-forming equation used here as a stand-in for mantle convection |
| GPE | gravitational potential energy (per unit area of a lithospheric column) |
| Euler pole | axis about which a rigid plate rotates on the sphere (Euler's rotation theorem) |
| Airy isostasy | each column floats independently at the height its density allows |
| flexural isostasy | columns coupled by bending stiffness, so loads spread over the flexural length α = (4D/Δρg)^¼ |
| Winkler foundation | restoring force proportional to vertical displacement, i.e. buoyancy |
| Cosserat medium | a continuum whose points carry orientation as well as position |
| polar decomposition | writing a deformation gradient as rotation × stretch |
| CF conventions | Climate and Forecast metadata conventions for gridded scientific data |
| HEALPix | Hierarchical Equal Area isoLatitude Pixelization of the sphere |
| downscaling, relaxation zone | regional climate modelling practice for driving a fine model with a coarse one at its boundaries |
