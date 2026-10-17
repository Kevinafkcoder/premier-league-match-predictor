# Premier League Match Predictor

## Data design
- 2024/25 + 2025/26: two completed seasons used as the historical baseline.
- 2026/27: current season. Its completed matches are used only for recent form (latest 5 fixtures).
- All three seasons live in one `data/matches.csv`.

## Updating the data
Run:

```bash
python data/update_data.py
```

This downloads completed Premier League results for all three seasons and rewrites `data/matches.csv`. Old matches are not deleted.

## Model flow
Historical attack/defence strength + current-season recent form + home advantage -> expected goals -> Poisson scoreline probabilities -> home/draw/away probabilities.

Backtesting is walk-forward on the current season, using only matches before each test fixture.
