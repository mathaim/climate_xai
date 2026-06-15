# Regional AR Intensity -> SAE Concept Organization — Design Spec

**Date:** 2026-06-15  **Status:** approved-for-planning

## Goal
Determine whether, where, and how GraphCast's sparse autoencoders organize
atmospheric-river (AR) *intensity*, across four landfall regions. SE Australia
is included deliberately as a different (Tasman/subtropical) AR regime to test
whether the SAEs distinguish it from the three westerly west-coast regions.

## Architecture
Staged pipeline with cached intermediates (each stage independently
re-runnable). Code in `src/analysis/ar_intensity/`. Outputs in
`/project/AikyamLab/madelyn/GraphCast/AtmosphericRivers/intensity_pipeline/`
(AR source masks remain on `/standard/.../AtmosphericRivers/Intensities/`).

## Locked parameters
- **Time alignment:** AR index `N` <-> activation timestep `N` <->
  `1979-01-01 00:00 + (N-1)*6h`, for `N = 1..56,700`. AR is a contiguous prefix;
  activations run to 2017-12-31 (~280 steps longer). 56,700 AR steps;
  56,980 activation slots (56,979 files, one corrupt L0 file deleted).
  Offset (00:00 vs +12h) is ASSUMED 00:00 and MUST be validated in Stage 3.
- **AR definition:** binary AR mask = `(class_masks == 2)`; background (0) and
  TCs (class 1) -> 0. (Not `event_masks != 0`, which includes TCs.)
- **Regions** (each 20 deg lat x 15 deg lon; cos-lat area-normalized):
  | Region | Lat | Lon |
  |---|---|---|
  | Western North America | 30 to 50 N | 230 to 245 E |
  | Western Europe | 37 to 57 N | 352 to 367 E (wraps prime meridian) |
  | Western South America | 30 to 50 S | 283 to 298 E |
  | Eastern Australia (Tasman) | 22 to 42 S | 147 to 162 E |
- **Coverage rule:** cos-lat area-weighted AR fraction of region >= 0.5, i.e.
  `sum(AR * cos(lat)) / sum(cos(lat))` over region cells.
- **IVT:** `IVT = (1/g)*sqrt((integral q*u dp)^2 + (integral q*v dp)^2)`,
  g=9.81, vertical integral over the full ERA5 column (upper-level q ~ 0, so
  bounds are not sensitive; equivalent to 1000-300 hPa). Per qualifying
  (timestep, region): MAX IVT over AR cells (`class_masks==2` within region).
- **Intensity bins:** per-region percentiles P10/P50/P90 -> {bottom10,
  low_mid40, up_mid40, top10}. Year-round; month/season retained as metadata.

## Stages

### Stage 1 — Relabel (inline)
Binary AR mask = `class_masks == 2`.

### Stage 2 — Regional coverage -> regional_coverage.parquet
For each timestep 1..56,700 x 4 regions: cos-lat area-weighted AR fraction;
`qualifies = fraction >= 0.5`.
Columns: time_index, datetime, region, coverage_frac, qualifies, month, season.
**CHECKPOINT (mandatory):** report qualifying counts per region (and a histogram
of coverage_frac). If counts are too low for stable percentiles/decoding (the
>=50% area rule is strict for a 20x15 box), revisit the threshold or pool
before proceeding. Do not build Stages 3-5 until counts are reviewed.

### Stage 3 — IVT intensity -> ar_intensity.parquet
For each qualifying (timestep, region): compute the 2D IVT field from ERA5
era5_inputs (q,u,v over pressure levels); take MAX IVT over AR cells in region.
Columns: time_index, datetime, region, max_ivt, season.
**Alignment validation:** on a sample, confirm AR cells overlap high IVT at
offset 0 vs +12h (2 indices); pick the offset with clearly better overlap and
record it. Downstream results are invalid until this confirms 00:00.

### Stage 4 — Per-region binning -> ar_intensity_binned.parquet + region_thresholds.json
Per region: P10/P50/P90 of max_ivt -> intensity_bin in {bottom10, low_mid40,
up_mid40, top10}. Save per-region thresholds.

### Stage 5 — SAE probing (all 6 SAEs)
SAEs and encode modes:
| SAE | checkpoint | layer activations | encode |
|---|---|---|---|
| Plain L0 | train/PlainSAE/Layer00/final_model.pt | /scratch/.../layer00 | x-mean,L2-norm; topk(relu) k=32 |
| Plain L8 | checkpoints/plain_layer8/final_model.pt | /project/.../Layer08 | same |
| Plain L15 | train/PlainSAE/Layer15/checkpoint_epoch008.pt | /project/.../Layer15 | same |
| Matry L0 | train/MatryoshkaSAE/Layer00/final_model.pt | /scratch/.../layer00 | feature_min/max [-1,1]; per_sample TopK |
| Matry L8 | train/MatryoshkaSAE/Layer08/final_model.pt | /project/.../Layer08 | same |
| Matry L15 | train/MatryoshkaSAE/Layer15/final_model.pt | /project/.../Layer15 | same |

For each qualifying AR: encode the SAE's layer activation -> latents
(40,962 x 4096); restrict to mesh nodes inside the region; per concept compute
the **mean activation rate** (fraction of region nodes with that concept active)
-> 4096-dim vector.

- **(A) Per-concept intensity profile:** mean of the 4096 vector across the 4
  bins, per region; surface intensity-tracking concepts (monotonic in IVT);
  cross-region (>=3 regions) and cross-SAE comparison.
- **(B) Decodability:** cross-validated linear probe predicting the 4-way
  intensity bin from the 4096 vector. Report **balanced accuracy** and the
  **majority-class baseline** (bins are 10/40/40/10). Include a **raw-512-dim
  activation baseline** (decode from the activation directly) to show the sparse
  code retains, not just inherits, the signal. Compare across all 6 SAEs and
  across layers (L0->L8->L15) and types (Plain vs Matryoshka).

## Grid note
AR coverage + IVT are on the 721x1440 lat/lon grid; SAE concepts are on the
40,962 icosahedral mesh nodes. Both restricted to the same region box, on
different grids; mesh nodes assigned to regions by their lat/lon.

## Dependencies to verify before implementation
1. Mesh-node lat/lons (map 40,962 SAE nodes -> region boxes); from GraphCast
   mesh definition.
2. era5_inputs structure: confirm q/u/v on pressure levels per timestep on the
   721x1440 grid, and the pressure-level coordinate for the vertical integral.

## Risks / open items
- **Strict coverage:** >=50% area of a 20x15 box is a very large AR; qualifying
  counts may be small. Stage-2 checkpoint gates this.
- **Alignment offset** assumed 00:00; Stage-3 validation must confirm.
- **SAE encode correctness:** Matryoshka must use per_sample + feature_min/max
  norm; Plain uses its own x-mean/L2 norm. Wrong mode invalidates Stage 5.
- **Compute:** encoding qualifying timesteps x 6 SAEs is the main cost (subset
  of 56,700; only qualifying ARs encoded).
- Region boxes equal in degrees, not area; analysis normalizes by cos-lat area.
