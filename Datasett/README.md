# Data layout

`master_dataset.csv` is a generated, mutable live input; `fpl.data.fetch` rebuilds it.
`frozen/master_dataset_gw231.csv` is the separate ignored research snapshot. Its tracked
admission manifest is `fpl/data/frozen_research_dataset_gw231.json`. Never replace the
frozen input with a refreshed live dataset.

The other CSVs and `R-Script 1 fetching data.r` are retained thesis-era source material.
The active Python ingestion path supersedes that script; those historical files are not
current setup commands. The old LSTM validation benchmark still used by diagnostics is
under `legacy/baseline_outputs/` and must remain available.
