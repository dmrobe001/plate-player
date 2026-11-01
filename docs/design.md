# Plate Player — design plan

Status: **phases 0–3 implemented** (2 and 3 partly, see the phase table), plate
identification from phase 5, Chebyshev acceleration, and the lava-lamp mantle with
grounded basal drag. **The physics is being redesigned around one pair-interaction
law** (sections 3.2, 4.0, 4.5, 4.7, 6): no bond/contact distinction, graded and
history-dependent integrity, area and volume as exact budgets. Of the redesign, phases
A (persistent edges), B (column profiles), C (the edge law, normal part), D (the
irreversible step, without slip), E (area, volume, melt, splits) and F (subduction,
suturing, slabs, arcs, bending) are done. Integration is next.
This document says where the model is going and why.

**Development strategy (current).** No more full-planet runs until all the physics
in the phase table is assembled. The observable phenomena we could calibrate
against (plate speeds, rigidity, ridge depth, orogen heights) are each produced by
several mechanisms at once, so tuning one component against them in isolation has
not converged. Until assembly, every component is developed and checked by **unit
tests** only, and every component stays **inspectable on its own**: see section 9,
"Development strategy".

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

3. **One pair-interaction law, no identities.** Every interaction between two
   columns is a function of the two columns' state and the state of the edge between
   them, and nothing else. There are no bond/contact kinds, and no plate or continent
   id enters any force. Bonds, contacts, faults, ridges and subduction interfaces are
   regions of one continuum (section 4.5). Plate ids and boundary classes are
   *diagnostics* derived from edge state and kinematics, for display, records and
   events.

4. **Exact budgets.** Surface area is exact by construction: each column's actual
   area is its cell in the tessellation of the markers, so the cells tile the sphere.
   Crust volume changes only at named, logged sources (melt at ridges, arcs, plumes)
   and sinks (slabs). Deformation moves area and volume between columns; it never
   creates or loses them. *(This replaces the earlier principle of a coarse
   plate-level solve, which phase 5 found incompatible with the drag-as-mass
   formulation; see "Phase 5 findings".)*

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
| `position` | vec3 | unit sphere | direction of the column |
| `elevation` | f32 | km | **free radius** (done in phase 3, as a separate coordinate rather than \|x\|; see phase 3 findings) |
| `director` | vec3 | unit vector | column axis; carries bending (new) |
| `orientation` | f32 | rad | rotation about the director, from the polar decomposition (new) |
| `area_ref` | f32 | unit sphere² | stress-free (reference) area; apothem a = √(area_ref / 2√3) sets rest lengths and contact size. Changed only by plastic/viscous dilatation (4.7) and splits |
| `cell_area` (derived) | f32 | unit sphere² | actual area: the column's cell in the tessellation of the markers (4.8; done in phase E). Recomputed each step, with `cell_area_prev`, `cell_shape` (centroid, major axis) and `cell_clipped` |
| `crust_volume` | f32 | km·unit sphere² | crust volume V; crust thickness h = V / area. Changes only at sources and sinks (4.8; done in phase E) |
| `thickness_km` (derived) | f32 | km | crust_volume / cell_area |
| `lith_mantle_thickness` | f32 | km | cold mantle layer that moves with the crust (new) |
| `crust_density` | f32 | kg/m³ | at mantle temperature, as now |
| `age` | f32 | Myr | time since formation (reset at breakup) |
| `thermal_age_myr` | f32 | Myr | sets the lithosphere, 2.32√(κt) capped by composition, for oceans and continents alike; scaled by stretching (done in phase E); reset by arcs (phase F) |
| `plastic_strain` | f32 | – | accumulated irreversible strain (diagnostic; weakening lives in edge integrity, 3.2) |
| `plate` | u32 | – | plate slot, from labelling; diagnostic only, never read by forces |
| `kind` | u32 | – | surface / slab / ghost (boundary-driven) / inactive |
| `level` | u32 | – | resolution level for adaptivity (new; 0 = base) |
| `strain` | 3×f32 | – | symmetric 2×2 strain from the neighbour fit (diagnostic + yield) |
| `stress` | 3×f32 | MPa·km | thickness-integrated stress tensor (diagnostic + export) |
| `x_prev`, `x_pred` | vec3 | | solver scratch |

Temperature is no longer a per-particle prognostic scalar. The column's thermal
structure follows from `thermal_age` (half-space cooling) plus a reheating rule for
slabs and arcs. That removes an explicit ODE and gives the depth–age law directly.

### 3.2 Edge = the interface between two neighbouring columns

There is one kind of edge. Whether a pair behaves like a weld, a fault, a gravel
contact or a ridge is a matter of the edge's state and the two columns' profiles
(section 4.5), not of a type field.

| Field | Type | Notes |
|---|---|---|
| `a`, `b` | u32 | particle indices; the pair is the edge's identity, and it persists across steps |
| `integrity` φ | f32 | 0..1. Scales cohesion, friction and (optionally) ductile viscosity. Lowered by plastic work, raised by thermally activated healing (4.7) |
| `length`, `slice_stress` | f32, 8×f32 | the pair's distance at the last update (strain increments are measured from it, so a change of rest length adds no stress), and the Maxwell stress of each of 8 depth slices of the shared interval: the memory that makes the edge history-dependent (4.5; phases C, D) |
| `strain`, `inelastic_strain`, `work` | f32 | diagnostics of the last update: strain vs rest length, inelastic strain (feeds area_ref), plastic work (feeds damage) |
| `stress_t` | vec2 | tangential stored stress (reserved; needs particle orientation, section 10) |
| `slip_anchor` | vec2 | tangential reference offset, moved by slip |
| `lambda`, `lambda_prev` | f32 | XPBD multipliers for this step (Chebyshev needs the previous one) |
| `timers` | f32 | sustained-compression accumulators for state transitions (subduction initiation, suturing) |
| `born` | f32 | Myr; for events and history |

**Existence is purely geometric.** An edge exists while the two columns are near in
map view (gap below a margin m·(aᵢ + aⱼ), m ≈ 0.1) *and* their depth intervals
overlap. It is created, with φ = 0 unless created by a split or by gap fill, when the
pair comes into range, and removed when it leaves range. Neighbour exchange (a pair
leaving range while another enters) is how the network flows in shear, as T1 events
do in a foam.

**Storage** (done in phase A). Edges live in one free-listed array and persist
across steps. Each particle has a slot list (≤12 edge ids, sorted by the partner's
angle in the particle's tangent plane); it serves the edge lookup, the tessellated
area (4.8) and the strain fit (4.2). The slot lists are an *index*, rebuilt every
step from the edge array, not state: snapshots hold only the edge array.

Boundary *classification* for display and export (divergent, convergent, transform)
is derived from the relative velocity across low-integrity edges, not stored.

### 3.3 Plate (diagnostic only)

Plates are connected components of high-integrity edges, labelled on the CPU every
0.25 Myr (section 5.3). They feed records, events and display, never forces. The
planned coarse rigid-body solve is shelved (phase 5 findings); its fields (`omega`,
`torque`, `drag_tensor`) are fitted from motion for export, not solved for.

| Field | Notes |
|---|---|
| `persistent_id` | stable across re-labelling (section 5.3) |
| `omega` | Euler-pole rotation fitted to member displacements over the labelling interval |
| `area`, `centroid` | for export |

### 3.4 Mantle

The mantle is a **basal-forcing generator** (section 8.4) on the CPU: the "lava lamp"
model in `src/mantle/`, described with its grounding in section 9, "Mantle and basal
drag". It supplies v_mantle(x, t) and its divergence; crust → mantle feedback
(blanketing, slabs as cold sinks) is not yet coupled.

---

## 4. Physics: one energy, one dissipation

Notation: A = column area, hₑ = elastic thickness (from `thermal_age` and crust
type), E = Young's modulus, ν = Poisson's ratio, g = gravity, ρ_m = mantle density.

### 4.0 Column profile (what each column contributes to its edges)

Each column carries a vertical profile, recomputed once per step from its own state
(surface temperature, thermal age, composition, crust thickness, lithosphere
thickness), on ~8 depth slices:

- **Temperature T(z):** oceanic, half-space cooling from `thermal_age`; continental, a
  steady geotherm with radiogenic heating in the crust. The lithosphere base Lᵢ is
  where T reaches ~1300 °C.
- **Density ρ(z)** from composition and T. Gives buoyancy (4.4), lithostatic pressure
  P(z) = ∫ρg dz, and GPE (4.6).
- **Ductile viscosity** η(z) = A exp(Q/RT(z)) (power-law creep linearised at the
  current strain rate); falls steeply with depth.
- **Brittle strength** τ_b(z) = μ(φ)·P(z) + φ·c (Byerlee friction plus cohesion);
  rises with depth.

The weaker of brittle and ductile at each depth is the **yield-strength envelope**
(brittle on top, ductile below). Elastic modulus E is taken as uniform. Everything
downstream reads these slices; nothing reads crust type directly, so a stretched
continent that turns oceanic changes behaviour through its profile alone.

*As built (phase B; `src/columns/profile.js`, `profile.wgsl`).* 8 slices: 3 over the
crust, 5 over the mantle lithosphere [crust, max(L, crust)]. Per slice: depth,
thickness, T, ρ, P, log₁₀η, ductile strength. Per column: L, Airy elevation (and the
unloaded freeboard), GPE, surface heat flow, Moho temperature, and the shallowest
brittle–ductile transition at φ = 1. Brittle strength needs φ, which belongs to the
edge, so the profile stores P and the edge law (phase C) applies μ(1 − λ_pore)P + φc.
Depth integrals (Airy, GPE, P) do not use the slices: they use 2-point Gauss–Legendre
quadrature on 6 crust and 16 mantle sub-intervals, down to the depth where T reaches
mantle temperature, so the slices only resolve what the edge law needs. Details,
parameter values and conventions: phase B findings.

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

*Role under the pair law (section 4.5):* the edges carry the in-plane mechanics. The
neighbour-fit tensor is kept as a measurement (display, export, and possibly a better
split of edge stress into isotropic and deviatoric parts than the per-edge stand-in
in 4.7), not as a separate energy, so elasticity is not counted twice.

### 4.3 Bending (new; needed at 50 km)

Directors nᵢ, with flexural rigidity D = E hₑ³ / 12(1−ν²):

> U_bend = Σ_edges (D Aᵢⱼ / 2ℓᵢⱼ²) |nⱼ − nᵢ − (rest difference)|²
>        + Σ_edges k_K (nᵢ · (xⱼ − xᵢ))²

The second term is a soft Kirchhoff constraint keeping the director normal to the
sheet. The discretisation is the least settled part of the design. It is accepted
only if it passes the line-load flexure benchmark (section 9) at 50 km.

*As built (phase F; `src/columns/flexure.js`, `flexure.wgsl`).* Not directors: the
elevation field itself is bent, with the tessellation's Laplacian.
- Energy per step: Σᵢ Aᵢ[(Kᵢτᵢ/2dt)(wᵢ − w₀,ᵢ)² + (Kᵢ/2)(wᵢ − w_eq,ᵢ)²] + Σᵢ (DᵢAᵢ/2) mᵢ²,
  with mᵢ = (1/Aᵢ) Σⱼ cᵢⱼ (wⱼ − wᵢ) and cᵢⱼ = shared cell boundary / distance (the
  finite-volume, or "cotangent", Laplacian of a Voronoi mesh). The first term is
  implicit Euler against the Haskell rebound time, the second the Winkler
  foundation (4.4), the third thin-plate bending (D/2)(∇²w)². The Gaussian-curvature
  part of the Kirchhoff energy only adds boundary terms and is left out.
- D = E hₑ³/12(1 − ν²), hₑ = Σ slices × their Maxwell factor this step: the same
  elastic fraction the edge law uses in-plane. `flexure_factor` scales D (0 = Airy).
- Solved by damped Jacobi sweeps (bending part of the diagonal × 2). With D = 0 one
  sweep is exactly the old per-column relaxation to Airy.
- Slabs are not in the network (they sink, section 6).

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

### 4.5 Pair interaction: the edge law

Given an edge (i, j) and the two column profiles (4.0), integrate over the depth
interval the two columns share. Depth slices act **in parallel** (columns are
vertically rigid: the thin-sheet assumption); each slice is a **Maxwell element**
(spring E in series with dashpot η(z)), integrated by implicit Euler.

**Geometry:** d = |xᵢ − xⱼ| on the unit sphere; rest length aᵢ + aⱼ (always: the
apothem sum); normal gap C = d − (aᵢ + aⱼ); tangential offset relative to
`slip_anchor`.

**Effective stiffness for this step** (normal and tangential alike):

> k = ∫_overlap dz / (1/E + dt/η(z)),  η from the pair (harmonic mean of the two profiles at each depth)

Cold slices act as springs, hot ones as dashpots; the **elastic thickness emerges**
rather than being a parameter. It is one number per edge per step, so the solve stays
linear within a step (the Chebyshev requirement). The stored stresses decay between
steps with each slice's Maxwell time η/E.

**Strength caps**, each the depth integral of the envelope over the overlap, applied
as clamps on the edge's multiplier λ (contacts already use exactly this, with a
tensile cap of 0):

| Direction | Brittle slices | Ductile slices |
|---|---|---|
| compression | μ(φ)·P(z) + φc (gravel resists compression) | flow stress |
| tension | φc only | flow stress (sign-blind: the plastic lower plate resists separation) |
| shear | μ(φ)·σ_n + φc | flow stress |

**Where load above a cap goes** (between solves, 4.7):
- the isotropic part (mean normal stress over a column's edges) changes the columns'
  `area_ref` at fixed crust volume: compression thickens, extension thins;
- the tangential part is slip: the anchor moves;
- the anisotropic normal part lets the pair separate or approach at the capped
  force, until it leaves range and another pair enters: neighbour exchange. The rule
  rest length = aᵢ + aⱼ stays exact.

**Limits of the continuum, by state, not by kind.** φ = 0 between cold, shallow
columns: tensile cap ≈ 0, a contact. High φ: a weld. Low φ under shear: a fault or
transform. Hot, thin, stretching columns: low caps everywhere, a ridge. A shrinking
depth overlap: a subduction interface (section 6).

*Prior art:* the bonded-particle model of rock mechanics in the discrete element
method (Potyondy & Cundall 2004), where bonds break into frictional contacts; here the
two are one edge with graded integrity.

*As built (phase C, normal part; `src/edges/law.js`, `law.wgsl`).*
- **Slices.** 8 equal slices over the shared interval (from the higher column top
  down to the shallower column base). At each slice, each column's temperature and
  creep strength are evaluated at its own depth (from its geotherm; pressure from its
  profile). Viscosities and ductile strengths combine by harmonic mean, pressures by
  arithmetic mean.
- **Memory is per slice** (8 stresses per edge), because one integrated number cannot
  decay correctly when cold and hot slices mix. A new edge loads every slice
  elastically from the rest length, σ = E·(strain now).
- **Regime and caps per slice.** A slice is brittle where friction (μ(1−λ_pore)P + φc)
  is weaker than creep, else ductile. Compression cap: the weaker of the two. Tension
  cap: φc if brittle, the flow stress if ductile (optionally reduced by lost integrity,
  `ductile_damage_weakening`, default 0; section 10).
- **Above a cap.** Ductile slices flow and brittle slices slip in compression: the
  stored stress is the capped stress. Brittle slices *crack* in tension: they store
  their elastic trial stress, less only the creep their capped stress drives, so they
  stay referenced to the rest length. Cracks close, and contacts engage, exactly at
  touching.
- **In the solve.** Per step, the slices give a linear law S = S₀ + K(ε − ε₀),
  with K = ΣE·r·dz and every slice included. XPBD (extended position-based dynamics)
  solves it as a spring with stiffness factor K/(E·h_ref) about a shifted zero-force
  length, with λ clamped to the depth-integrated caps. A cracked pair is therefore
  pulled together at most at its tension cap until it reaches touching, where the law
  is stiff again. An earlier draft treated open cracks as constant forces for the
  whole step, and pairs overshot into compression.
- **Not built: the tangential (shear) part.** A tangential spring between column
  centres resists rigid rotation of the pair (the bug removed in phase 2). It needs a
  per-particle orientation (4.2) first (section 10). Until then shear is resisted only
  through the network's geometry, as before.

### 4.6 Gravitational potential energy (ridge push, plateau collapse)

A thin-sheet column model loses the horizontal force from lateral differences in
gravitational potential energy (GPE). Add it as an explicit, smooth body force:

> Fᵢ = −Aᵢ ∇Γ,  Γ = ∫ (lithostatic pressure − reference column pressure) dz

The gradient comes from a least-squares fit over neighbours. With depth–age
cooling this produces ridge push, and on thick continents it produces plateau
collapse and orogenic spreading. Explicit is acceptable here because Γ varies
slowly; a test should confirm it has no stability limit in practice.

*As built (phase D; `columns_gpe_force` in `flow.wgsl`, reference `flow.js`).* The
least-squares gradient of Γ over the column's listed neighbours (start-of-step
positions), applied after the edge update as a shift of the predicted position,
Δx = −∇Γ·h·dt/η_a (the column's basal drag; independent of its area). Scaled by
`gpe_force_factor`, **default 0**: against today's drag, a column at a continental
margin (ΔΓ ≈ 2×10¹² N/m over one spacing) is pushed at metres per year, and a planet
run with it on breaks the stability tests (phase D findings). Calibrated at integration.

### 4.7 Irreversible processes (between solves)

All are applied once per step, after the solve, from converged stresses. This keeps
the solve linear within a step and means solver residuals below yield (~MPa) change
nothing permanent.

- **Edge stress:** σᵢⱼ from λ and the stored stress. Each column's mean pᵢ over its
  edges is the isotropic part; sᵢⱼ = σᵢⱼ − ½(pᵢ + pⱼ) stands in for the deviatoric
  part until the strain fit (4.2) gives a tensor.
- **Volumetric flow (per column):** with p̃ᵢ = pᵢ + (excess GPE of i relative to its
  neighbours)/Lᵢ, if |p̃ᵢ| exceeds the column's integrated yield,
  d ln area_ref/dt = sign(p̃ᵢ)(|p̃ᵢ| − Yᵢ)/ηᵢ. Crust volume is unchanged, so
  thickness changes and isostasy (4.4) moves the elevation. The GPE term is what
  makes thick columns spread under their own weight (plateau collapse); without it
  mountains have no ceiling. This is the thin viscous sheet of England & McKenzie
  (1982), whose Argand number (buoyancy stress / viscous stress) decides between
  thickening and collapse.
- **Slip:** tangential load above the shear cap moves `slip_anchor`.
- **Integrity:**

  > dφ/dt = −Ẇ_p / W_D + (1 − φ)/τ_heal(T)

  Ẇ_p is the plastic work rate on the edge (cap-limited force × plastic
  displacement rate, all directions). It is positive for compression, extension and
  shear alike, so any permanent deformation reduces integrity. τ_heal is thermally
  activated: hot material heals fast. This mirrors grain-damage theory of plate
  boundary formation (Bercovici & Ricard 2012). W_D and τ_heal are the two parameters
  that decide whether damage localises into narrow boundaries; they are calibrated at
  integration, not before.
- **Thermal:** `thermal_age` advances; stretching and melt addition reheat
  (reduce thermal age in proportion), as do arcs.
- **State transitions on edges** (need persistent edges): subduction initiation and
  suturing via the edge timers (section 6).

*As built (phase D; `src/edges/law.js`, `src/columns/flow.js`, `law.wgsl`, `flow.wgsl`).*
- **Edge memory is the pair's distance** at the last update. The rest length (from
  area_ref) then only sets where new edges start and where columns are in range.
  Changing it creates no stress, so area_ref change is an outlet, not a double count.
- **Inelastic strain** of an edge in a step: the strain increment its *stiffest*
  slice (largest Maxwell factor r) did not store elastically. A column shortens
  permanently only as fast as its strongest layer lets it.
  - A cold lid below yield holds the column, however much the hot base creeps.
    Stress relaxing below is load moving to the lid, not shortening.
  - A lid at its cap lets the column flow.
  - An all-hot column flows by its coldest part.
  - Crack opening is not inelastic: it closes again.
  - An elastic-weighted mean over slices was tried first and failed: a pair held
    still drifted in area as its mid slices relaxed.
- **Volumetric flow** replaces the p̃ > Y rule above:
  d ln area_ref = 2 × mean of the column's edges' inelastic strain, at fixed crust
  volume (thickness ∝ 1/area_ref until phase E's V/area). The GPE term reaches it
  through the GPE force (4.6) loading the edges, not as a separate pressure.
  - Uniform shortening at the caps gives d ln A = 2 dε.
  - In pure shear the per-edge parts cancel: tension cracks and compression stays
    elastic, so no area changes.
- **Plastic work**: cap × plastic strain × dz, summed over slices at a cap, plus
  cohesion × opening for cracks. Creep below a cap is not counted.
- **Integrity**: φ ← max(φ − W/W_D, 0), then 1 − φ decays by exp(−dt/τ_heal).
  - τ_heal = heal_time × exp(Q_h/R (1/T − 1/T_m)).
  - T is the temperature of the edge's load-bearing part: slice temperatures
    weighted by r·dz.
  - Defaults: W_D = 50 MPa·km; τ_heal = 0.1 Myr at mantle temperature, 53 Myr at
    1200 K, never below ~900 K.
  - This replaces the stand-in yield damage and the stand-in healing kernel.
- **Slip: not built.** It needs the tangential channel (4.5, As built).

### 4.8 Area, volume and melt

**Area is geometric.** A column's actual area is its tessellation cell: the polygon
bounded by perpendicular bisectors to its angle-sorted neighbours (≤12), computed per
particle. The cells tile the sphere, so Σ area = 4π exactly and coverage cannot leak.
What used to be a void is now a column whose cell is stretched: it thins
(h = V/area), subsides, floods (water loading), and produces melt. Passive rifting
needs no special rule. `area_ref` (the apothem) is the stress-free size; the gap
between the two is elastic dilatation, which drives the volumetric flow of 4.7.

**Volume sources.** Passive decompression melting (McKenzie & Bickle 1988), driven by
the geometric dilatation rate (the local divergence of the marker motion) plus active
upwelling from the mantle model:

> V̇ᵢ = areaᵢ · w · F̄ · max(1 − Lᵢ/H_m, 0),  w = H_m · max(ȧreaᵢ/areaᵢ, 0) + w_mantle,ᵢ

- H_m ≈ 70 km is the depth where rising mantle crosses its solidus; F̄ ≈ 0.1 the mean
  melt fraction. H_m·F̄ = 7 km is why steady ridges make 7 km crust at any spreading
  rate.
- Under thick lithosphere (L ≥ H_m) stretching only thins; melt begins once the
  lithosphere is thin enough. That is continental breakup, and the first column to
  cross into oceanic composition is logged as the breakup event.
- w_mantle (plumes, with a hotter, deeper H_m) adds volume at fixed area: hotspot
  plateaus and seamounts. Volcanism is a volume source; a ridge is an area source.
- Arcs: overriding columns near a subduction interface gain volume in proportion to
  the subduction flux (section 6).

**Volume sinks:** slab columns leaving the surface (their volume is logged when they
become slabs and when they are removed).

**Splits.** A column whose actual area exceeds twice the base area splits along its
direction of greatest stretch; parent and child share volume and area_ref; the edge
between them starts at full integrity, and edges to former neighbours are reassigned
by side. This replaces the spreading proxy (inflation over diverging mantle), which
is removed.

*As built (phase E; `src/columns/area.js`, `src/columns/volume.js`, `area.wgsl`,
`split.wgsl`).*
- **Cells are weighted, not plain bisectors.** Between columns i and j the boundary
  is the great circle in the plane x·(aⱼpⱼ − aᵢpᵢ) = 0 with a = 1/cos(apothem).
  - Between touching columns it passes exactly through the contact point; between
    equal columns it is the perpendicular bisector.
  - The normals of any three such planes sum to zero, so the planes share a line and
    the cells meet at common vertices: the cells tile the sphere exactly, whatever
    the weights. (A spherical cousin of the power diagram.)
- **Each cell is computed alone**, in the gnomonic projection about its column, where
  great circles are straight lines. The cell is a convex polygon: a regular 12-gon of
  radius `cell_clip_radius` (0.04) clipped by one half-plane per candidate
  (Sutherland–Hodgman). Its spherical area is the sum of fan triangles (Van
  Oosterom–Strackee formula). Everything is in the column's own tangent coordinates,
  so float32 is enough.
- **Candidates are all columns in the 27 hash cells around**, not the ≤12 listed
  edges. Edges exist only within the range margin, so a column at an opening gap
  would lose the neighbours that bound its cell. The hash scan reaches at least one
  hash cell (0.094), and the clip radius is kept below half of that, so every
  column whose boundary can cut the clip polygon is seen.
- **Clipped cells.** A cell still touching its clip polygon has a free side (nothing
  within reach): there the tiling is not exact. Such cells are counted, and they never
  split (their area is not a measure of stretching).
- **Thickness is derived:** h = V / cell area, every step. At load, V is set from the
  loaded thickness × the tessellated area.
- **Lithosphere thinning uses thermal age.** Every column, oceanic or continental,
  has a lithosphere 2.32√(κt) capped by composition (100 km oceanic, 150 km
  continental). Per step t ← min(t, cap)·(A₀/A)², which thins the lithosphere by
  A₀/A, McKenzie's (1978) uniform stretching factor β. It then recovers by ageing:
  a continent re-thickens to 150 km in ~130 Myr.
- **Melt** per step: dV = F·H_m·max(A − A₀, 0)·max(1 − L/H_m, 0), with L the thinned
  lithosphere. Melt (`melt_density` 3000) is mixed into the column's composition by
  volume. A continental column crossing 2850 kg/m³ this way has broken up: it is
  counted and its formation age reset.
- **Plumes (w_mantle) and arcs are not built.** The mantle model has no plume source
  yet; arcs are phase F.
- **Splits** use the actual cell area (> `split_area_factor` 2 × the base hexagon).
  The halves go to the cell centroid ∓ 0.87 × the cell's major-axis spread
  (√ of its largest second moment). That puts them on the two half-cells' centroids
  for a domino of hexagons (0.885), a rectangle (0.866) or an ellipse (0.849).
  - Parent and child share volume and area_ref equally.
  - The child copies the parent's material (no fresh oceanic crust).
  - The edge between them starts intact and stress-free.
  - Every edge touching either half restarts its strain measure from the new
    geometry (its `length`), keeping its stored stresses, so a split adds no stress.
  - Neither half counts dilatation (melt, thinning) in the step after the split.
- **Volume budget:** sources (melt) and sinks (columns becoming slabs) are summed on
  the GPU in 64-bit fixed point (10¹² units per km·unit²), since load. Slab
  assimilation no longer touches the surface budget.
- **Removed:** the spreading proxy (area growth over diverging mantle) and its three
  parameters.

---

## 5. The time step

### 5.1 Schedule

Target order once the pair law is assembled (today's code runs stage 1, stage 2
including the edge law's normal part, stages 3–4, stage 5 without slip, and stages
6–7; the GPE force is implemented but off by default;
see phase A–E findings). As built, splits run at the start of the step on last step's
cells, and the tessellation (with melt) runs after the hash grid is built, before the
profiles:

```
1. profiles                   per column: T(z), ρ(z), η(z), envelope (4.0); tessellated area (4.8)
2. edges                      hash grid; create pairs coming into range, remove pairs leaving range;
                              per edge: effective stiffness k and caps for this step (4.5)
3. predict                    x̃ᵢ = xᵢ + v_mantle(xᵢ)·dt + F_ext,i·dt/γᵢ
4. solve                      Chebyshev-accelerated Jacobi XPBD over all edges, λ clamped to caps
5. irreversible               edge stress; volumetric flow; slip; integrity; melt; thermal;
                              edge state transitions; splits; slab removal (4.7, 4.8)
6. isostasy                   elevation relaxes toward Airy height (4.4)
7. finish                     v = (x − x_prev)/dt (diagnostic); recording if due;
                              plate labelling every 0.25 Myr (CPU, diagnostic only)
```

Each numbered stage reads and writes named buffers only, so any stage can be run
alone on dumped or synthetic inputs (section 9, development strategy).

### 5.2 XPBD with drag in place of mass

Multiplying Φ by dt gives ½|x − x̃|²_Γ + dt·U. For a constraint Cₖ with compliance αₖ
(energy Cₖ²/2αₖ):

> Δλₖ = (−Cₖ − α̃ₖ λₖ) / (Σᵢ wᵢ |∇ᵢCₖ|² + α̃ₖ),   Δxᵢ = wᵢ ∇ᵢCₖ Δλₖ
> with **wᵢ = 1/γᵢ** (instead of 1/mᵢ) and **α̃ₖ = αₖ/dt** (instead of αₖ/dt²)

Velocity is no longer state; it is a diagnostic. Large dt is safe; it only makes
each step more nearly quasi-static.

### 5.3 Plate labelling (diagnostic)

Plates are connected components of edges with integrity above a threshold, found
exactly by union-find on the CPU every 0.25 Myr; ids persist by majority inheritance,
which yields split/merge/lost events (phase 5 findings). Under the pair law, two
plates either side of a ridge stay joined by low-integrity, yielding edges, so the
labelling cuts at low integrity (or high plastic work rate), not at missing edges.
Labels never feed back into forces.

### 5.4 Parallelism and determinism

- Jacobi sweeps with per-iteration fixed-point integer atomics (WGSL has no float
  atomics); no graph colouring needed.
- Thread-timing-dependent behaviour (free-list pops, edge creation order) is listed
  in record manifests, not engineered away. Runs should agree statistically, not bit
  for bit.

---

## 6. Boundaries, subduction and spreading

Under the pair law (4.5) boundaries are not built; they are where edge state and
column profiles have become weak. This section says how each kind arises.

**Rifting and spreading.** Where columns are pulled apart, the tension is resisted by
the ductile slices whatever the integrity, so separation is taken up by stretching:
tessellated area grows, crust thins at fixed volume, the column subsides and, below
sea level, floods. Stretching reheats, and hot is weak, so strain localises: the
positive feedback by which real rifts narrow. Once the lithosphere is thinner than
the melting depth, new area arrives with melt (4.8) and the column becomes oceanic:
breakup. Steady ridges make ~7 km crust; split columns become the new sea floor. Ridges
stand high because young lithosphere is thin and hot (4.4), and ridge push follows
(4.6).

**Shear and transforms.** Pure shear does not change a column's area, so it cannot be
taken up by volumetric flow; it loads edges tangentially until slip (anchors move)
and integrity loss, then neighbour exchange. Low-integrity edges under sustained shear
are transforms and strike-slip faults.

**Subduction initiation.** An edge's timer accumulates while its normal stress is
compressive beyond a threshold; it initiates when the timer passes a window *and* at
least one column is negatively buoyant (old ocean, ≳30–40 Myr). Polarity: the denser
column goes down; if both qualify, the older. Buoyant (continental) columns never go
down; continent–continent convergence thickens instead.

**Subduction (explicit slab).** The downgoing column's `kind` becomes slab. It
keeps its edges to the trailing plate, so slab pull is transmitted through real
edges. It sinks under its own negative buoyancy, bends via the director energy, and
its depth overlap with the overriding plate shrinks, so the edge law's caps on that
interface fall and they slide past each other (a low-integrity, weak interface). At a cutoff depth (e.g. 660 km), or once
thermally assimilated, the particle is removed, its volume is logged as a sink, and its edges are dropped. The slab is
the memory: polarity persistence and the slab-pull–speed feedback come from the
geometry, not from bookkeeping.

**Arcs.** Overriding columns within a set distance of a subduction interface gain crust
volume at a rate proportional to the subduction flux, and their `thermal_age` is
reset (hot, weak arc lithosphere). This is how continents grow over time.

**Collision and suturing.** Buoyant columns in sustained compression thicken through
volumetric flow (4.7), bounded by GPE spreading. Integrity across the collision zone
heals while the zone is compressed and quiet, which welds the plates (suturing); the
plate labelling records the merge.

*As built (phase F; `src/edges/transitions.js`, `law.wgsl`, `thermal.wgsl`,
`flexure.wgsl`).*
- **One timer per edge** (in `edges_maxwell`, after the stresses): + dt while the
  depth-averaged normal stress is below −`transition_stress_mpa` (50 MPa), reset to 0
  otherwise.
- **"Negatively buoyant" = freeboard < 0** (the column's Airy height without water
  is below the datum): the whole column is denser than the mantle. From the column
  profile this is ocean older than ~8 Myr (not 30–40: the basaltic crust's
  compositional buoyancy is small at 3000 kg/m³); calibrate at integration.
- **Polarity:** the column touching a slab, if only one can sink and does; else the
  lower freeboard; ties to the lower index. Window `subduction_window_myr` (1 Myr),
  × `subduction_propagation_factor` (0.1) when the down column already touches a slab:
  the trench propagates along the plate and keeps its polarity.
- **Suturing:** both buoyant, timer past `suture_window_myr` (5 Myr), and no plastic
  work on the edge that step: integrity → 1.
- **The overlap stand-in is gone** (edges_create no longer turns deeply overlapping
  columns into slabs; `subduction_overlap` removed).
- **Slabs** keep their place and elevation when they subduct, then sink at
  w = B g (2a) / 4η with B = −ρ_m × freeboard (kg/m²), η the upper-mantle viscosity
  (`isostatic_viscosity_pa_s`), a the apothem: ~7 cm/yr for old ocean. They warm
  (thermal age × exp(−0.05/Myr · dt)), so they slow, and are removed at
  `slab_cutoff_km` (660) or at neutral buoyancy. Their volume leaves the surface
  budget when they subduct.
- **Slab pull through edges:** an edge with a slab end is measured in 3D (radius
  1 + elevation/R), so a sinking slab stretches it and pulls the plate behind it. Its
  edges restart their length measure when it subducts (no stress jump). The edge
  keeps existing while depth intervals overlap (a slab ~100 km down detaches).
- **Weak interface:** compression caps on edges with exactly one slab end × 0.1
  (`slab_interface_weakening`). The overburden of a slab is mantle rock, not water.
- **Arcs:** a slab crossing the arc window (80–160 km) supplies
  `arc_efficiency` (0.2) × its volume over the passage, shared equally among surface
  columns within `arc_reach` (0.05, ~320 km) on the far side from the plate behind
  the slab (the integrity-weighted direction to its surface edge partners), mixed in
  at `arc_density` 2750 (so arcs build continental crust); their thermal age resets
  to 0. Logged as a source.
- **Not built:** the slab's vertical load on the plate behind it (the trench and
  outer rise). Bending exists (4.3), but the slab–plate coupling is only in-plane.

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

### Development strategy: assemble, then run

**Why.** The phenomena available for calibration are coupled: plate speed depends on
basal drag, edge strength, slab pull and ridge push at once; rigidity on solver
convergence and on yield; ridge depth on cooling, isostasy and melt. Tuning one
component against such observables while the others are missing or stand-ins has
repeatedly moved the problem rather than solved it (spreading proxy, overlap trigger,
contact runaway). So:

1. **No full-planet runs until assembly.** Planet benchmarks resume only when every
   phase up to "Integration" in the table is in. Earlier findings below stay as
   history; their numbers describe the old physics.
2. **Unit tests only until then.** Each component is checked against an answer
   known without the rest of the model: an analytic solution, a conservation
   identity, a limit case, or a reference implementation. Tests stay cheap
   (seconds), as in `test/`.
3. **Every component inspectable on its own.**
   - Each physics component has a **pure CPU reference** in JavaScript (a function
     from plain arrays to plain arrays) next to its WGSL kernel. Unit tests check the
     reference against the analytic answer and the kernel against the reference.
   - Each pipeline stage (5.1) reads and writes named buffers, and can be **run alone**
     from a state dump (snapshot or synthetic), so a suspected fault can be tested
     on harvested data without warming up a planet (README, "Debugging").
   - Intermediate per-column and per-edge quantities (profiles, k, caps, φ, plastic
     work, melt rate, tessellated area) are exposed through a debug readback, not
     only their effect on positions.
4. **Calibrate at integration, all at once**, against the coupled targets listed in
   the Integration row, and record the result once (no regression tests of
   high-level behaviour).

### Phase table

Each phase ends with a check that could fail. Until integration these are
mechanism tests (cheap, small, one mechanism with a known answer; kept in `test/`).
Benchmarks of high-level behaviour are run once, at integration, and their numbers
written here.

| Phase | Work | Acceptance test |
|---|---|---|
| 0 ✅ | Housekeeping: exponential integrators for all rates; cache nearest mantle node; fix free-list and state-flip races; mantle substep from a stability bound, mantle time tied to Myr | Every slot free or active exactly once; cached node = nearest; no blow-up at 10× dt. dt sensitivity is *reported*, not asserted: PBD stiffness still depends on dt until phase 2, so that number is the baseline phase 2 must beat |
| 1 ✅ | **Minimal recording:** manifest + field series + snapshots (tiers a–c), download as zip, resume from zip | Restart from a record ZIP in a fresh page continues the run (bitwise at the time; now checked to a tolerance, since bit-identity is not a goal) |
| 2 ✅ (partly) | **Overdamped XPBD core:** drag as mass, real compliance. *Changed:* λ-consistent Jacobi instead of graph-coloured Gauss–Seidel (see below) | Two-particle relaxation matches implicit Euler, (1 + dt/τ)⁻ⁿ with τ = γ/2k, to 0.1% at dt/τ = 0.1, 1, 10 (kept as a test). Whole-planet dt-independence **not met**: blocked on plate-scale convergence (findings below) |
| 3 ✅ (partly) | **Unified edge array + column contacts + free radius & buoyancy.** Done: bonds and per-step one-sided contacts in one edge array solved by one XPBD kernel; contacts need overlap in map view *and* depth; fully damaged bonds retire; the radial coordinate is free and relaxes to Airy height (Haskell rebound time). *Superseded in part* by the pair law: edge kinds and per-step contacts give way to phase A | Thickened block relaxes to its Airy height as implicit Euler with the rebound time, neighbours unmoved (kept as a test); columns touch only when they overlap in depth (kept as a test) |
| 5 (identification ✅; coarse solve shelved) | **Plate labelling** (diagnostic), Euler-pole fits, events | Labelling, persistent ids, split/merge events and Euler-pole fits are unit-tested on synthetic networks. The coarse *solve* does not fit the XPBD formulation (findings below) |
| A ✅ | **Persistent pair-keyed edges**, one kind: per-particle angle-sorted slot lists; create on entering range (map view + depth), remove on leaving; φ, stored stresses, anchor, timers carried across steps | Slot lists stay symmetric (i lists j ⇔ j lists i) under random motion; an edge's state survives 100 steps of motion within range and is dropped exactly when the pair leaves; no duplicate pairs; free-list invariants hold. **Met** (kept as tests; phase A findings) |
| B ✅ | **Column profile** (4.0): T(z) from thermal age / continental geotherm, ρ(z), η(z), envelope; lithosphere thickness; GPE | Oceanic profile matches half-space cooling; Airy elevation from the profile reproduces depth ∝ √age on a synthetic age ramp; envelope peak at the brittle–ductile transition where the two curves cross (analytic). **Met**, plus GPE differences = half-space ridge push, and the kernel matches the reference (kept as tests; phase B findings) |
| C ✅ (normal part) | **Edge law** (4.5): effective stiffness integral, caps as λ clamps, Maxwell stored-stress decay | Single edge under constant separation rate: stress approaches η_eff·rate with the Maxwell time (implicit-Euler sequence, exact); cold pair behaves as a spring, hot pair as a dashpot; tension cap of a φ = 0 cold pair is ≈ 0; two-particle relaxation test from phase 2 still passes when k is set from a uniform profile. **Met** for the normal part (kept as tests; phase C findings). Tangential part deferred: needs particle orientation |
| D ✅ (without slip) | **Irreversible step** (4.7): volumetric flow at fixed volume, slip, integrity (work-driven damage, thermal healing), neighbour exchange; plus the GPE force (4.6) | Uniform compression of a patch thickens at the predicted rate with Σ volume constant; pure shear leaves every area_ref unchanged and produces slip; φ decays in proportion to work input and heals with τ_heal(T); a patch below yield accumulates no plastic strain over many steps with solver noise. **Met**, with "slip" read as crack opening and plastic work, since edges have no tangential channel; neighbour exchange is the range rule of phase A (kept as tests; phase D findings) |
| E ✅ | **Area, volume, melt, splits** (4.8); remove the spreading proxy | Tessellated areas sum to 4π to float precision on random and lattice configurations; a stretched strip produces 7 km crust per unit new area at L = 0 and none at L ≥ H_m; a split conserves volume and area_ref; budget identity: Δ(Σ V) = logged sources − sinks exactly. **Met** ("exactly" = to the float32 rounding of each column's V; kept as tests; phase E findings) |
| F ✅ | **Subduction and suturing as edge-state transitions**; slab particles, bending (4.3; elevation on the tessellation, not directors); arcs | Timer-based initiation fires on a synthetic convergent pair with a dense side and never with two buoyant sides; line-load flexure profile within 10% (a mechanism test on a strip, not a planet). **Met** (kept as tests; phase F findings) |
| **Integration** | Assemble A–F; first planet runs since the redesign | Calibrate together, recorded once: plate speeds (rms ~4 cm/yr, slab-attached plates fastest), interior rigidity (non-rigid strain rate ≲ 10⁻⁹/yr), ridge depth ∝ √age, ~7 km oceanic crust, orogen heights bounded by collapse, coverage exact, volume budget closed, damage localised into narrow boundaries |
| 9 | **Adaptivity** (refine first, coarsen second) | Benchmarks of Integration give the same continuum answer at 220/110/50 km within stated tolerances |
| 10 | **Patch mode:** run L1 code on a projected patch driven by a Forcing; adapters from records; event log | Patch driven by a global record stays within tolerance of the global fields in its interior over [t₀, t₁] |

*Mapping from the earlier plan:* old phase 4 (neighbour-fit strain) is folded into D
as a measurement (4.2); old 6 (lithosphere column) is B; old 7 (subduction) is F; old 8
(volume conservation, collision, suturing, arcs) is split between D, E and F.

Done outside the numbered phases: Chebyshev acceleration of the Jacobi sweeps, the
lava-lamp mantle with grounded basal drag, and bond rest length = apothem sum
(findings sections below).

**Resolved:** how new crust is made. Not by inflation inside plates, and not by a
separate gap-filling rule: by stretching (tessellated area grows at fixed volume) plus
decompression melt where the lithosphere is thin (4.8).

### Phase F findings: subduction, suturing, slabs, arcs and bending

**What was built** (rules: 4.3 and section 6, "As built").
- **Edge timer and transitions** at the end of `edges_maxwell`; CPU reference
  `src/edges/transitions.js`.
- **Slabs** (`thermal.wgsl`): sink at a Stokes-like rate from the profile's
  freeboard, warm, and are removed at 660 km or at neutral buoyancy. Edges with a slab
  end are measured in 3D (`edge_offset`, bindings.wgsl) and lubricated in
  compression. The slab's overburden in its profile is rock, not water.
- **Arcs** (`slabs_arc_count`, `columns_arc` in `flexure.wgsl`).
- **Bending** of the free radius (`columns_flexure_begin`, `columns_moment`,
  `columns_flexure`); CPU reference `src/columns/flexure.js`. It replaces the
  per-column Airy relaxation in `crust_thermo_isostasy`.
- **Removed:** the overlap stand-in for subduction (`subduction_overlap`), the slab
  pull velocity in `crust_predict`, the fixed slab drop and sink rate
  (`slab_initial_depth_km`, `slab_sink_rate_km_myr`), and crust assimilation
  (`slab_assimilation_km_per_k`).
- **New particle fields:** `slab_neighbours`, `geometry_reset`, `arc_count`,
  `arc_supply`, `slab_trailing`, `flexural_rigidity`, `bending_moment`,
  `elevation_start`. The slot-list record per particle grew to 44 words (the
  tessellation neighbours and bending weights). `plate.loadState` accepts `state`.

**Tests** (kept). Unit tests of the references:
- **Timer and transitions:** the timer counts only beyond the threshold and resets;
  old ocean against a continent goes down only past the window; of two oceans the
  older goes down; a column touching a slab goes down on a tenth of the window; two
  buoyant columns never subduct, and weld after the suture window only when quiet.
  Freeboard: +0.30 km at 2 Myr, −0.14 at 10, −1.27 at 60; continent +3.41.
- **Slab sinking:** old ocean sinks at 72 km/Myr (7.2 cm/yr); buoyant columns don't.
- **Flexure on a strip of 55 tessellated columns** (α = 109 km = 3.4 spacings):
  peak 0.1506 km against 0.1485 analytic, first zero at 2.34α (3π/4 = 2.36),
  forebulge 4.1% of the peak (e^(−π) = 4.3%), worst deviation 1.4% of the peak. The
  GPU iteration converges to the exact solve; with D = 0 one sweep is exactly the old
  Airy relaxation.

GPU tests:
- **Edge timers** (four pairs held at ±0.2% strain, elastic): old ocean against a
  continent subducted at step 5 (window 4.5 steps); the continent pair and the young
  ocean pair welded at step 10 (window 9.5); the pair in tension kept timer 0.
- **Slab** (a row of five bonded old-ocean columns, the last a slab, and a continent
  beyond it): sinks 1.16 km per step, matching the reference to 6×10⁻⁷; at 55 km its
  edge carries 1.8 MPa of tension and the far end of the plate has moved 8.9 km toward
  it (slab pull through the edge); removed at the (test) cutoff; the arc got
  exactly 20% of its volume, all to the continent, none to the plate behind.
- **Flexure on the GPU** (the same strip, D from a 50 km elastic profile): D exact;
  deflections within 0.03% of the CPU exact solve and 1.5% of the analytic profile.

**Found by the tests.**
- **`interface` is a reserved word in WGSL**: the whole compute module failed to
  compile, so every test stalled rather than failed. Page errors are checked only at
  the end of each test; when many tests stall at once, look at the page errors first.
- **Arcs fed the slab's own plate**, resetting its lithosphere hot (L = 0), which ended
  the depth overlap and detached the slab. Arcs now grow only on the far side from
  the plate behind the slab.
- **Removed slabs were still in the step's hash grid** (built before slabs sink), so
  `edges_create` could bond to them; found by the 10× time-step planet test. Hash
  scans now skip inactive particles.
- **Arc heat makes melt:** a column whose lithosphere was reset by an arc melts where
  its cell grows (decompression), so arcs and rifting interact. Reasonable; noted.

**On a generated planet, measured once:** no slab forms in 20 steps (0.32 Myr; the
window is 1 Myr). The 10× time step (1.6 Myr) does subduct; it passes with p99 speed
6.8 cm/yr, max 25 (2.6 and 23 before phase F). Bending deepens the deepest columns by
~0.2 km. The default GPU suite now takes ~5.5 minutes under SwiftShader.

### Phase E findings: area, volume, melt and splits

**What was built** (rules: 4.8, "As built").
- **`columns_area`**: each surface column's weighted Voronoi cell, from the hash grid.
- **`columns_volume`**: dilatation since last step thins the lithosphere (thermal
  age) and brings decompression melt; volume, composition, thickness = V / area.
- **`columns_split`** (replaces `crust_accumulation`): splits on the actual cell area,
  along the cell's major axis, stress-free.
- **`columns_init_volume`**: V = loaded thickness × cell area.
- **New particle fields:** `crust_volume`, `thermal_age_myr`, `cell_area`,
  `cell_area_prev`, `cell_shape`, `cell_clipped`, `volume_added`; `surface_area` is
  renamed `area_ref`. The solver buffer's header grew to 12 words (clipped cells,
  volume budget, breakups). The record schema is 0.5.0.
- **Lithosphere from thermal age for continents too** (capped at 150 km), so that
  stretching can thin it. The CPU references (`lithosphereThicknessKm`, `geotherm`)
  changed with it.
- **`?grid=N`** sets the hash grid (coarse test worlds need larger cells).

**Parameters:** `cell_clip_radius` 0.04, `melt_depth_km` 70, `melt_fraction` 0.1,
`melt_density` 3000, `split_area_factor` 2 (now on the actual area; was 3 on
area_ref).

**Tests** (kept). Unit tests of the references:
- **Tiling:** cells sum to 4π to 5×10⁻¹⁴ on an icosphere lattice, and on jittered
  configurations with apothems varied ±20% and ±40%.
- **Contact point:** the boundary between touching columns passes through it.
- **Cell shape:** a 2s × s rectangle has its long axis along x and spread 2s/√12; a
  split puts the halves at ±s/2 to 1%.
- **Melt:** a ridge column stretched to twice its area makes exactly 7 km of crust
  per unit new area and stays 7 km thick; old ocean and a continent stretched 10%
  make none, and their lithosphere thins by 1/1.1; L = H_m/2 makes half.
- **Breakup:** a continent stretched 1% per step starts melting at β = 2.15 (L just
  below 70 km), and crosses into oceanic density at β ≈ 10.6, with 6.6 km of crust.

GPU tests, on a whole sphere of 2562 columns (jittered icosphere, apothems ±15%,
`?capacity=4096&grid=20`):
- **Tiling:** Σ cell areas − 4π = −1.2×10⁻⁷ (relative 10⁻⁸); each cell matches the
  reference to 7×10⁻⁷; none clipped; V = h × area at load.
- **Melt, one step after a smooth displacement:** 454 stretched young columns made
  7.0000 km of crust per unit new area; old ocean and continent made none; each column
  matches the reference exactly.
- **Budget:** Δ(Σ V) − (sources − sinks) = −3×10⁻⁸ against a float32 rounding bound of
  10⁻⁶. The logged sink equals the volume of the one column pushed into a slab.
- **Splits** (342 cells above 1.15 × base): Σ V and Σ area_ref unchanged; each pair
  of halves equal; the parent–child edge intact with stress below 0.03 MPa (float32
  position rounding); halves at the reference positions to 5×10⁻⁵ spacings (WGSL's
  sin and cos are accurate only to 2⁻¹¹); after the split step the cells still sum to
  4π to 4×10⁻⁸; edge storage invariants hold.

**Found by the tests.**
- **Isolated columns split.** In small test patches, columns with neighbours on one
  side only have clipped cells several times the base hexagon, and split at once.
  Clipped cells now never split.
- **Thermal age must be clamped before it is scaled.** A continent loaded with age
  1000 Myr took dozens of steps of stretching before its lithosphere started to thin.
- **The coarse test sphere starts with overlaps** above `subduction_overlap`, so the
  overlap stand-in for subduction (phase F) made dozens of slabs. The tests set it
  out of reach.

**On a generated planet, measured once** (20 steps, default parameters):
- **Coverage:** cells sum to 4π − 0.08%. The deficit is in ~213 clipped cells (2%) at
  voids in the initial state wider than the hash reach.
- **No melt in 20 steps.** The initial oceans are 60 Myr old (L = 100 km), so they must
  be stretched by β > 1.4 before melting starts. There are no ridges in the initial
  state; making new sea floor depends on stretching by the mantle. Seeding young
  ridges is an integration question.
- **Cells next to voids take the void's area**, so their crust thins (7 → 4.8 km in
  5 steps): what used to be a gap is now a stretched column, as intended. Only 29
  splits happened, because the large cells there are clipped.
- **Breakup needs β ≈ 10.** Mixing melt by volume and calling a column oceanic past
  2850 kg/m³ means melt must outweigh the original crust. Real margins break up at
  β ≈ 3–5; a rule on the melt's share of the crust's top, or on the lithosphere, may
  be better. Decide at integration.
- No capacity overflows; the free-list test and the 10× time step pass (p99 2.6,
  max 23 cm/yr).

### Phase D findings: the irreversible step

**What was built** (rules: 4.6 and 4.7, "As built").
- **`edges_maxwell`** now does more after advancing the slices. It records each edge's
  inelastic strain and plastic work, accumulates the inelastic strain on both columns,
  and applies the integrity law.
- **`columns_flow`** turns the accumulated inelastic strain into area_ref and
  thickness changes.
- **`columns_gpe_force`** pushes columns down the GPE gradient (off by default).
- **Removed:** the stand-in yield damage, the stand-in healing kernel, and their
  parameters (`yield_stress_mpa`, `activation_energy`, `reference_temp`).
- **Changed edge fields:** edges now remember their length rather than their strain.
  The record schema is 0.4.0.

**Parameters** (provisional, calibrated at integration):
- `damage_work_mpa_km` 50, `heal_time_myr` 0.1, `heal_activation_kj` 250;
- `gpe_force_factor` 0 (off; see below);
- `volumetric_flow_factor` 1. The storage tests set it to 0, and it is a tuning
  handle.

**Tests** (kept). Unit tests of the references:
- inelastic strain is 0 below yield and the full increment at the cap;
- plastic work = cap × plastic strain;
- a crack opening is not inelastic but does work against cohesion;
- in pure shear of a hexagon of edges the area is unchanged, and uniform shortening
  at the caps gives d ln A = 2 dε at constant volume;
- integrity falls by exactly W/W_D and heals as exp(−dt/τ) with the Arrhenius ratio;
- the least-squares gradient is exact for linear fields.

GPU tests, on a 19-column patch moved by hand:
- **Uniform shortening, 0.4% per step:** the crust thickens by ln 0.0479 over
  steps 6–12 against 0.0481 predicted, and total volume holds to 10⁻⁵.
- **Integrity** matches 1 − ΣW/W_D to 2×10⁻⁷.
- **Pure shear:** no area changes at all. Work appears on the 14 stretched edges and
  on none of the other 28.
- **Below yield, solved for 30 steps:** zero work, integrity exactly 1, zero area
  change.
- **GPE push:** matches the reference to 2×10⁻³.

**Found by the tests.**
- **Inelastic strain** had to come from the stiffest slice, not a weighted mean
  (4.7, As built).
- **The GPE kernel read neighbours' predicted positions** while they were being
  pushed (a race); it now reads start-of-step positions.
- **Load transfer is real.** In a patch strained to ~1 MPa, the hot base relaxes and
  passes its load to the 47 km lid, which roughly doubles the lid stress. A patch
  built flat on the sphere starts pre-stretched ~3×10⁻⁴ (25 MPa), enough to crack.
  The test declares its loaded state stress-free, as the generated planet is.

**GPE force against today's drag, measured once.**
- **A free column at a ridge-like age gradient** (9–51 Myr across 900 km) is pushed
  at 141 cm/yr.
- **On a generated planet with the force on**, the first 3 steps give median speed
  ~30 cm/yr, p99 120–160 cm/yr, and up to 35 m/yr at poorly connected margin
  columns. Mean integrity falls 0.39 → 0.18 as the push breaks edges.
- **With the force off**, median speed is 5 → 3 cm/yr, p99 30 → 11 cm/yr, and
  integrity settles near 0.87.
- **The physics is right; the balance isn't.** ΔΓ ≈ 2×10¹² N/m across a margin is
  realistic, but basal drag (η_a = 10¹⁹ Pa·s over a ~200 km channel) is too weak to
  hold it, and broken plate edges carry no tension.
- **So the force is off by default.** Turning it on is the first calibration at
  integration, together with drag, slab pull and edge strength.

**Planet runs, once** (force off): the free-list test passed, and the 10× time step
reached p99 3.0 cm/yr, max 19.2.

### Phase C findings: the edge law (normal part)

**What was built.** Two kernels around the solve. `edges_law` runs after the edge
update and gives each edge, from its slices and stored stresses, this step's
stiffness factor, zero-force gap and λ bounds. `edges_maxwell` replaces the old stress
rule: it advances and caps each slice's stress and stores it. The solver kernel lost
its integrity special cases; it solves the linear law with λ clamped. The CPU
reference is `src/edges/law.js`. Rules: section 4.5, "As built".

**Parameters.**
- `elastic_thickness_km` (30) is replaced by `stiffness_reference_km` (150). It only
  normalises: edge stiffness is E times the elastic part of the shared interval, and
  dividing by the thickest column keeps each edge's factor ≤ 1, so the Chebyshev
  spectral-radius bound, computed for factor 1, stays valid.
- Consequence: a fully elastic 100 km pair is 3.3× stiffer than the old 30 km bond.
- New: `ductile_damage_weakening` (0).
- Column extent: an edge's depth interval, and edge existence, now use
  max(lithosphere, crust) as the column's depth. Newborn crust (lithosphere 0) used to
  have no depth extent and no edges.

**What the law gives, measured once** (CPU reference, default parameters):
- **Old ocean pair** (100 Myr, 100 km shared):
  - elastic thickness Σ r·dz = 47 km: the top is elastic, the base flows;
  - compression cap 10 600 MPa·km;
  - tension cap 2430 MPa·km at φ = 1, of which cohesion is 750.
- **Same pair at φ = 0:** tension cap 1680 MPa·km. That is the ductile base's flow
  stress, which the design table lets resist separation whatever φ. It is about 80% of
  ridge push (~2100 MPa·km at 50 Myr), so a broken boundary between old plates would
  resist opening nearly as hard as an intact one. With `ductile_damage_weakening` = 1
  it is 0. This is the decision in section 10.
- **Fully cold pair at φ = 0:** tension cap exactly 0.

**Tests** (kept). Unit tests of the reference:
- the stress under constant separation rate equals the implicit-Euler Maxwell sequence
  to 10⁻¹², and reaches η·rate after 10 Maxwell times;
- a cold pair holds its stress for 100 steps (a spring); a hot pair's stress is η × the
  step's rate and is gone a step later (a dashpot);
- old ocean is elastic on top and viscous below;
- a cold φ = 0 pair carries no tension, closes exactly at touching and pushes past it;
  crushed beyond its cap it sits at the cap and unloads elastically.

GPU tests:
- the kernels match the reference on 6 real-profile pairs (old ocean, continents,
  continent against ocean, thick against thin, φ = 1, 0.3, 0) to ≤ 10⁻⁶ of scale;
- the Maxwell sequence under imposed separation matches to 4 digits;
- the phase 2 relaxation and Chebyshev chain tests pass unchanged with a uniform
  elastic profile (factor 1).

Planet runs, once: the free-list test, and the 10× time step (finite, p99 speed
5.3 cm/yr, max 19.9).

**Noticed for later.**
- **Linear in a step, slices capped after it.** The λ clamp acts on the total, while
  each slice is capped only after the solve. Within a step, the total can sit under
  its cap while one slice is over its own; the next step corrects it.
- **Viscosity is linearised** at the reference strain rate (10⁻¹⁵/s). Power-law
  creep at the edge's actual rate is a refinement for integration.
- **Damage and healing are still the stand-ins**, now acting on the mean slice
  stress. Phase D replaces them with the work-driven integrity law.

### Phase B findings: column profiles

**What was built.** A `ColumnProfile` per particle (a derived buffer, never saved),
written every step by `column_profile` before isostasy, and its CPU reference. Isostasy
now targets the profile's Airy height; nothing else reads the profile yet.

| Piece | Choice | Why |
|---|---|---|
| Oceanic geotherm | half-space cooling, T = Tₛ + ΔT erf(z / 2√(κt)); L = 2.32√(κt) (the 90% isotherm); age capped where L = 100 km (~59 Myr) | the existing L, and a cheap stand-in for the plate model's flattening |
| Continental geotherm | steady conduction, heat production 1.5 µW/m³ in the top 15 km, T(L = 150 km) at the 90% isotherm, gradient continued below L to mantle temperature | gives 41 mW/m² surface heat flow (cratonic) and a 334 °C Moho under 35 km of crust |
| Density | composition density at mantle temperature × (1 + α(Tₘ − T)) in crust and mantle | unchanged convention |
| Airy | freeboard = ∫(1 − ρ/ρₘ)dz down to where T = Tₘ; water-loaded below sea level | whole column, so the lithospheric mantle's contraction sets depth vs age |
| GPE | Γ = g ∫(ρ − ρ_ref)(r − r_b)dr, ρ_ref = mantle below the datum and 0 above, at the Airy height | independent of the base r_b in equilibrium (tested), so no compensation depth parameter |
| Brittle | τ = φc + μ(1 − λ_pore)P, μ = 0.6, λ_pore = 0.4 (hydrostatic), c = 20 MPa | Byerlee friction |
| Ductile | dislocation creep at ε̇ = 10⁻¹⁵ s⁻¹: wet quartzite (Gleason & Tullis 1995), diabase (Mackwell et al. 1998), dry olivine (Hirth & Kohlstedt 2003); shear strength = σ_diff/2, η = strength/2ε̇ | standard laboratory laws. Constants are from memory of the literature and should be checked against the papers before calibration; order-one factors between uniaxial laws and tensor invariants are ignored |
| Brittle–ductile transition | shallowest slice midpoint where ductile < brittle, then bisection | a diagnostic; the edge law integrates the envelope itself |

**Tests** (kept). Unit tests of the reference against analytic answers:
- the oceanic geotherm satisfies ∂T/∂t = κ∂²T/∂z², and the continental one kT″ = −A
  with continuous flux;
- for a crustless column, subsidence below the ridge is
  2ρₘαΔT√(κt/π)/(ρₘ − ρ_w) (Turcotte & Schubert) to 2×10⁻⁶ for ages 1–150 Myr;
- GPE differences equal the half-space ridge push,
  gρₘαΔTκt[1 + 2ρₘαΔT/π(ρₘ − ρ_w)], to 3×10⁻⁷ (4.2×10¹² N/m at 100 Myr);
- moving the integration base 100 km deeper changes GPE by ≤ 1.2 MPa·km (ridge
  push is ~2000);
- the envelope peaks where the curves cross: olivine at ~650 °C, quartzite at ~260 °C.

GPU: the kernel matches the reference on 11 columns (ridge to old ocean, crustless,
shelf to plateau) to ≤ 2×10⁻⁵ of each field's scale. The Airy-block test (phase 3)
now targets the profile's Airy height. One planet-scale run (5 steps): no errors; the test
took 66 s instead of 56 s under the software GPU.

**Behaviour changes.**
- Hypsometry: ridge 2.85 km deep, old ocean 5.62 km, 35 km continent +0.81 km
  (Earth: ~2.5, ~5.7, ~0.8). This needed sea level at 2.6 km in the elevation datum
  (was 3.5); 30 km continental crust now sits just under water, as a shelf.
- Elevation no longer depends on the per-particle `temperature` scalar, which is
  left for the stand-in yield and healing rules (and slab heating) until phases C–D.

**Noticed for later.**
- Continents are "jelly sandwiches": the strongest point is brittle upper mantle at
  ~70 km, below a weak quartz lower crust. The edge law integrates the whole
  envelope, so it will see this, and continental stiffness will depend on the Moho
  temperature.
- At age 0 the column is at mantle temperature throughout (no lithosphere, no
  brittle layer), and the reported heat flow uses a 1 kyr floor on age.
- Sea level is the one calibrated number here; it moves with crust densities and
  thicknesses at integration.

### Phase A findings: persistent edges of one kind

**What was built.**
- One `Edge` struct and one free-listed array (131 072 slots); bonds and contacts are
  gone. Integrity in [0, 1] replaces both kinds and damage: an edge at integrity 0 pushes
  only (λ ≥ 0), one above 0 is two-sided with tension softened by integrity. Initial
  bonds load at 1 − damage; edges made later start at 0, except a split's parent–child
  edge (1). The stored-stress, slip-anchor, timer and born fields are carried but not
  yet used.
- Per step, on predicted positions: prune pairs out of range (or with a dead particle)
  and list the survivors in both ends' slot lists; each particle i creates edges to
  in-range partners j > i that it does not already list; then each slot list is sorted
  by angle. No pair is ever created twice: pair (i, j) is only created by thread i,
  after checking i's survivors, and edges that other threads append to i's list
  during creation have partners below i. New slots are popped with a compare-and-swap
  loop; prune (pushes) and create (pops) are separate dispatches.
- CPU reference (`src/edges/topology.js`): the range rule, the update
  (kept/created/removed), the angle basis, and a checker for the storage invariants,
  also exposed in the page as `plate.debug`.

**Tests** (kept): unit tests of the reference (lattice degree 6 and 60° slot angles;
range boundary at exactly the 10% margin; depth exclusion; keep/create/remove; deep
overlap creates no edge but keeps an old one). GPU: 127 columns under a random walk for
30 steps (181 created, 204 removed) match the reference every step with no storage
problems; 7 columns with written-in edge state keep it bit for bit over 100 steps of
jitter; pulling one column out drops its centre edge at step 7 and its ring edges at
step 12, as the reference predicts; moved back, the pairs get fresh edges (integrity 0,
no history, born now). One planet-scale run (5 steps, full capacity): invariants hold.

**Behaviour changes, by design:**
- A cohesive edge stretched beyond 10% is dropped (it was a bond before, kept at any
  length). The two-particle relaxation test now stretches by 5%.
- A new slab's edges leave range when its top drops below the lithosphere, so slabs
  detach from their plates until phase F.
- Rewiring at a split keeps the edge's integrity (it used to reset damage to 0,
  welding the rewired bond).
- Old record ZIPs (with `constraints` buffers) no longer restore; the record schema is
  0.2.0.

**Test cost.** Under the software GPU the cost of a dispatch or readback scales with
buffer capacity, and a readback takes ~0.5 s, so a 100-step planet test took tens of
minutes. Mechanism tests now run on an empty page with 1024 slots (`?empty&capacity=1024`)
and write inputs without readbacks; the default GPU suite takes ~2 minutes. Planet
tests run only with `PLATE_PLANET=1`.

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

*Deferred to integration* (development strategy above): these are planet runs,
and the solver's load changes once caps and the effective stiffness integral are in.

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
- **Measured after the change** (same benchmark: seed 1, 8 warm-up + 16 measured steps;
  the refitted initial state starts with 9.5% damaged bonds):

  | configuration | median plate speed | non-rigid, all | non-rigid, interiors | damaged bonds |
  |---|---|---|---|---|
  | Chebyshev 32, growth on | 3.1 cm/yr | 1.60 cm/yr | 0.64 cm/yr | 10.0% |
  | Chebyshev 32, growth off | 0.6 cm/yr | 0.27 cm/yr | 0.11 cm/yr | 9.5% |

  Tearing is mostly gone: growth now adds 0.5 points of damage over the window instead
  of 1.7. Deformation is not gone, and that is expected rather than numerical:
  inflation *is* a prescribed areal strain rate, about the upwelling divergence
  (~1%/Myr), and a plate ~1000 km across that inflates at 1%/Myr has edge velocities
  relative to a rigid fit of ~10 km/Myr = 1 cm/yr, the size measured. Inflation also
  drives the plates (median speed 0.6 -> 3.1 cm/yr), which is a growth artefact, not
  mantle drag. So the remaining question is not a solver one: crust creation inside
  plates is an unphysical strain source, however it is accommodated. The options in
  the decision above still apply; the rest-length rule stays either way.

### Phase 3 findings: unified edges, column contacts, free radius

**What was built.**
- *Edges.* Bonds keep their slots [0, maxBonds), managed by the free list. Contacts
  fill [maxBonds, maxConstraints) and are rebuilt every step: after the hash build,
  each touching, unbonded pair that overlaps in depth gets a CONTACT edge. Bonded
  pairs are skipped using a per-step list of bond partners (≤12 per particle). The
  same XPBD kernel solves both kinds: the separate collision projection, with its
  averaging, is gone. Contacts carry no cohesion and are never damaged. A bond that
  reaches full damage is retired, and the pair interacts by contact from then on.
- *Column contacts.* A column spans [top − lithosphere thickness, top] in the
  elevation datum. A new slab's top drops to 160 km down and sinks at 20 km/Myr
  (stand-ins until phase 7). So slabs pass beneath surface columns instead of being
  exempted by a special rule, and the old slab "basal friction" rule is gone.
- *Free radius.* `elevation` is now state: each step it relaxes toward its Airy height
  by implicit Euler with the Haskell rebound time τ = 4πη/(Δρ g λ), with λ = 4 apothems
  and η = 10²¹ Pa·s (upper mantle, from post-glacial rebound). That gives τ ≈ 28 kyr
  against dt = 16 kyr. The water-loading factor is corrected to ρ_w/(ρ_m − ρ_w).

**Changed from the plan: the radius is a separate coordinate.** The plan put the free
radius in |x|. But in-plane strain must then be measured on directions anyway: a column
1 km higher has a chord 1/6371 longer, which reads as 11 MPa per km of relief at
70 GPa. A separate radial coordinate keeps the horizontal solve on the unit sphere
and loses nothing until bending couples the two (phase 7).

**Contacts must be one-sided and found with a margin.** The first version froze each
step's contacts at the start and made them act both ways, which keeps the iteration
linear for Chebyshev. On the planet it ran away: by 2.4 Myr, unbonded overlaps
reached 46%, p99 speed 200 cm/yr and median |σ| 240 MPa. Hypothesis: the solve pushes
particles into neighbours that had no contact edge at the step start. Tested: every
overlap deeper than 5% (14 at 0.48 Myr, 55 at 0.96 Myr) was between a pair *without*
a contact edge. Fix: create contacts for gaps below 10% of the apothem sum, and clamp
each contact's multiplier at λ ≥ 0 (push only). Chebyshev tolerates this nonlinearity
so far. Same probe, seed 1:

| | t | unbonded overlap p99 / max | speed median / p99 | median / p99 \|σ\| | bonds | damaged |
|---|---|---|---|---|---|---|
| frozen two-sided | 0.96 Myr | 7.9% / 20% | 2.6 / 77 cm/yr | 113 / 807 MPa | 22 303 | 4.7% |
| one-sided + margin, growth on | 0.96 Myr | 1.1% / 1.8% | 2.2 / 7.1 cm/yr | 114 / 643 MPa | 22 928 | 4.1% |
| one-sided + margin, growth off | 0.48 Myr | 0.11% / 0.17% | 0.8 / 2.4 cm/yr | 16 / 81 MPa | 23 962 | 3.4% |
| one-sided + margin, growth off | 0.96 Myr | 0.15% / 0.26% | 0.4 / 1.5 cm/yr | 20 / 106 MPa | 23 962 | 3.4% |
| before phase 3 (growth on) | 0.96 Myr | 2.2% / 3.4% | 3.0 / 52 cm/yr | 99 / 772 MPa | 25 581 active, 17.7% damaged | |

(Initial bonds: 25 581; those starting at full damage are retired in the first step,
so "damaged" counts only bonds still active.)

**What this says.**
- With growth off the model is steady: no new damage, overlaps under 0.3%, about
  20 MPa median stress.
- With growth on, stress climbs about 60 MPa per 0.5 Myr, with or without phase 3. The
  climb is the in-plate inflation, now fully transmitted because bonds and contacts
  agree on the rest length. Interpenetration and runaway speeds were phase 3's own
  problem, and are fixed.
- **Subduction no longer initiates.** The trigger is an overlap above 0.012 (34% of an
  apothem sum), and stiff one-sided contacts keep overlaps near 1%. With growth off,
  converging plates lock and slow (median speed 0.8 → 0.4 cm/yr). The overlap trigger
  was always a proxy; design §6 calls for a stress-based initiation rule (sustained
  compressive normal stress on a contact with a negatively buoyant side), which needs
  persistent contacts. That moves forward from phase 7.
- Healing no longer re-welds fully broken bonds, because they are retired. Suturing
  (phase 8) is the intended replacement.

**Spreading proxy, regrounded.** Crust over diverging mantle grows in area at the
divergence rate (above 0.002/Myr, capped at 5%/Myr). Upwelling divergence is ~1%/Myr,
so global crust production is ~0.5%/Myr of the surface, against Earth's ~0.6%/Myr
(~3 km²/yr). This is still a proxy until phase 8 ties creation to where plates actually
separate.

## 10. Open questions

- **Localisation:** does work-driven damage with thermal healing concentrate into
  narrow plate boundaries, or weaken plates diffusely? Bercovici's group finds
  localisation only in a window of healing rates. Decide W_D and τ_heal at
  integration.
- **Neighbour exchange at 50 km:** do capped forces plus range-based edge turnover
  give sensible ductile shear zones, or does the deviatoric part need a per-edge
  outlet (a zero-mean rest offset, which would amend rest = aᵢ + aⱼ)?
- **Ductile tension at zero integrity** (phase C): tuned at integration with the
  other interaction parameters (`ductile_damage_weakening`). New hot edges are held
  by their surroundings and heal fast; columns made in a highly dynamic setting
  should not cohere (transform faults cut ridges).
- **Shear on edges** (phase C): a tangential channel with slip anchors needs a
  per-particle orientation (the polar decomposition of the neighbour fit, 4.2) so
  rigid rotation stays free. Not scheduled; revisit if integration shows it matters.
- **GPE force vs basal drag** (phase D): the force is physical but, against today's
  drag, metres per year at margins. Calibrate drag, edge tension and this force
  together at integration.
- **Caps inside Chebyshev:** clamping λ to caps is a nonlinearity like the contact
  clamp, which has been tolerated so far. Check at integration that dense yielding
  zones do not slow or break convergence.
- **Subduction calibration** (phase F): freeboard < 0 makes ocean older than ~8 Myr
  subductable (design intent 30–40 Myr); the 50 MPa threshold, 1 Myr window and the
  interface weakening are guesses. Set them at integration with plate speeds.
- **Trench and outer rise** (phase F): the slab's vertical load on the plate behind
  it is not coupled to bending; the flexure solver could take it as a line load.
- **Bending at fine resolution** (phase F): Jacobi needs ~(α/ℓ)⁴ sweeps (~2000 at
  α/ℓ = 3.4). Free at today's 220 km (α/ℓ < 1); at 50 km it needs Chebyshev or
  multigrid.
- **Tessellated area on the sphere** (resolved in phase E): weighted cells from the
  hash neighbourhood tile exactly; the only leak is at voids wider than the hash
  reach (clipped cells, counted).
- **Breakup criterion** (phase E): volume-mixed composition crosses into oceanic only
  at β ≈ 10. Decide at integration whether breakup should key on the melt's share of
  the upper crust or on lithosphere thickness.
- **Initial ridges** (phase E): the generated planet's oceans are all 60 Myr old, so
  nothing melts until stretched by β > 1.4. Seed young ridges at integration?

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
| integrity φ | an edge's undamaged fraction: scales cohesion, friction and optionally ductile viscosity |
| thin viscous sheet | lithosphere modelled as a depth-averaged viscous layer (England & McKenzie 1982); Argand number = buoyancy stress / viscous stress |
| downscaling, relaxation zone | regional climate modelling practice for driving a fine model with a coarse one at its boundaries |
