# Multiscale Education Diagnostics

## Decision question

At what spatial scale do county-level social gradients in educational attainment
operate? A single global relationship may conceal effects that are local,
regional, or effectively statewide.

## Workflow

1. Infer the missing projected CRS by comparing candidate UTM coordinates with
   supplied longitude and latitude fields.
2. Fit global OLS and diagnose its residual spatial dependence with Queen
   contiguity weights and Moran's I.
3. Fit adaptive-kernel GWR, then fit MGWR so each covariate can select its own
   neighbourhood scale.
4. Report local coefficient maps, corrected significance, and an external GWR4
   replication check.

## Evidence delivered

In addition to coefficients and fit statistics, `outputs/results.json` reports
the GWR AICc gain over OLS, the reduction in absolute residual Moran's I, and
the range of MGWR bandwidths. These make the spatial-scale argument directly
auditable.

## Run

```bash
python analysis.py
```
