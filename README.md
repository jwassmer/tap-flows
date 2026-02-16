# tap-flows

Traffic Assignment Problem (TAP) tooling for:
- user equilibrium and social optimum flows on directed graphs,
- multicommodity flow optimization,
- social cost gradient analysis,
- figure generation for synthetic and OSM-based city networks.

## Repository Layout

- `src/`: core library modules (optimization, graph generation, plotting, OSM processing).
- `test/`: automated tests.
- `analysis/`: canonical scripts used to generate publication figures.
- `experiments/`: ad-hoc scratchpad/trash folder for informal experiments and one-off scripts.
- `data/`: cached graph inputs and experiment artifacts.
- `figs/`: generated outputs (PDF/PNG/GeoJSON).

## Quick Start

1. Create and activate an environment (Python 3.9+).
2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Run tests:

```bash
python3 -m pytest -q
```

## Common Usage

Single-commodity TAP:

```python
import numpy as np
from src import TAPOptimization as tap

G = tap.random_graph(num_nodes=20, num_edges=40, seed=42, alpha="random", beta="random")
P = np.zeros(G.number_of_nodes())
P[0] = 500
P[1:] = -500 / (G.number_of_nodes() - 1)

f_ue = tap.user_equilibrium(G, P)
f_so = tap.social_optimum(G, P)
```

Multicommodity TAP:

```python
import numpy as np
from src import multiCommodityTAP as mc

G = mc.random_graph(num_nodes=15, num_edges=30, seed=42, alpha="random", beta="random")
od = -np.ones((G.number_of_nodes(), G.number_of_nodes()))
np.fill_diagonal(od, 0)
np.fill_diagonal(od, -np.sum(od, axis=0))
demands = [od[:, i] for i in range(G.number_of_nodes())]

flow = mc.solve_multicommodity_tap(G, demands)
```

Generate figures:

```bash
python3 analysis/fig_classic_braess_social_cost.py
python3 analysis/fig_classic_braess_scgc_validation.py
python3 analysis/fig_user_equilibrium_vs_social_optimum.py
python3 analysis/fig_multicommodity_decomposition.py
python3 analysis/fig_synthetic_gamma_scan.py
python3 analysis/fig_cologne_stadium_traffic.py
python3 analysis/fig_potsdam_braess_osm.py
python3 analysis/fig_lambda_inequality_diagram.py
```

## Notes

- Some figure and OSM workflows need additional geospatial dependencies and external data.

## License

No license file is currently provided. Add `LICENSE` before publishing publicly.
