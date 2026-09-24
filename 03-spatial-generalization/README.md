# Spatial Generalization Stress Test

## Decision question

How much apparent map accuracy remains when a model is evaluated away from its
nearest training samples? The Meuse floodplain provides a controlled comparison
of interpolation-oriented and process-informed models.

## Workflow

1. Fit a spherical variogram to log zinc and use Dutch RD New (EPSG:28992) for
   metre-based spatial operations.
2. Compare ordinary kriging with random forests using process covariates, with
   and without raw coordinates.
3. Evaluate random k-fold, coordinate-blocked k-fold, and buffered
   leave-one-out validation.
4. Map final covariate-driven predictions on the supplied grid.

## Evidence delivered

`outputs/results.json` reports the RMSE penalty from spatial blocking and from
a 1 km buffer for every model. This turns the usual random-versus-spatial CV
comparison into an explicit transferability stress test.

## Run

```bash
python analysis.py
```

Buffered validation performs thousands of fits and takes a few minutes.
