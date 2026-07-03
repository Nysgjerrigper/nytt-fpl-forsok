# FPL prognosemodell + MILP-optimaliserer

Opprinnelig en masteroppgave (LSTM-prognoser i R + MILP-lagvalg i Python).
Nå omskrevet til én Python-pipeline for faktisk ukentlig bruk i sesong,
i tillegg til fortsatt metodisk etterprøvbar.

## Mappeinnhold
- **fpl/**: hele den aktive pipelinen.
  - `config.py` — stier og sesongparametre.
  - `data/fetch.py` — henter og renser FPL-data fra vaastav sin GitHub-database (dynamisk sesong-/GW-deteksjon, ingen hardkodede sesonger).
  - `features.py` — rullerende form-features per spiller (erstatter LSTM-ens lærte embeddings).
  - `model/train.py`, `model/predict.py` — per-posisjon LightGBM-modeller (GK/DEF/MID/FWD), walk-forward-validering.
  - `milp/optimize.py` — konsolidert MILP-lagvalg (budsjett, formasjon, kaptein, transfers, chips), basert på Kristiansen et al.-formuleringen.
  - `run_week.py` — ukentlig kjøreskript: oppdater data → tren modeller → hent kommende fixtures fra offisielt FPL API → optimaliser lag/transfers.
- **Datasett/**: rådata og `master_dataset.csv` (generert av `fpl/data/fetch.py`).
- **legacy/**: original masteroppgave-kode (R-prognosemodell/LSTM, de 8 opprinnelige MILP-variantene, gamle valideringsprediksjoner) — beholdt som dokumentasjon av selve oppgaven, ikke i aktiv bruk.

## Bruk
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m fpl.data.fetch          # bygg/oppdater master_dataset.csv
python -m fpl.model.train         # tren modeller + skriv ut evaluering mot gammel LSTM

python -m fpl.run_week --team-id <ditt FPL-lag-ID> --horizon 3
```
Uten `--team-id` gir `run_week.py` en "bygg fra scratch"-anbefaling i stedet for å videreføre et eksisterende lag.
