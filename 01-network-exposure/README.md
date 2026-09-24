# Network Exposure Audit

## Decision question

Would a public-health accessibility conclusion change if distance is measured
along the walkable network rather than as a straight line? This study uses the
Soho cholera data as a network-exposure audit rather than a historical map
replication.

## Workflow

1. Reproject source geometry to British National Grid (EPSG:27700) for metric
   distances.
2. Repair street topology, create a weighted graph, and snap pumps and death
   addresses to its edges.
3. Compare Euclidean assignment with shortest-path assignment.
4. Test the observed concentration against 999 random household placements on
   the same street network.
5. Produce KDE, service-area, and null-distribution figures.

## Evidence delivered

`outputs/results.json` records both the core allocation result and three audit
metrics: the share of cases whose allocation changes, the observed network
concentration relative to the null mean, and the observed walk-distance
reduction relative to the null.

## Run

```bash
python analysis.py
```

The included figures are regenerated under `figures/`.
