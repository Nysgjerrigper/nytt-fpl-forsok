# Training Ground operations: Ownership control — completed schedule

Status: **COMPLETED**; evidence: `solio_schedule`.

Unranked. Points, differences and paired intervals are not assessed.

Display preserved operational evidence without paired scoring, ranking or promotion.

## Recorded activity

| Observation | Value |
| --- | --- |
| requested_origins | [137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152] |
| completed_origins | [137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152] |
| saved_origins | 16 |
| failed_origin | Not available |
| failed_stage | Not available |
| recorded_error | Not available |
| Counting scope | VENDOR_WRAPPER_INVOCATIONS |
| Attempted solver calls | 16 |
| Returned solver calls | 16 |
| Solver calls that raised | 0 |
| Calls with no recorded outcome | 0 |
| Saved origin decisions | 16 |

## Timing evidence

Seconds, within each declared boundary; overlapping timings must not be added.

| Boundary | Recorded evidence |
| --- | --- |
| admission_seconds | {"count": 16, "not_measured": 0, "observations": 16, "total": 2.4300070848839823} |
| preparation_seconds | {"count": 16, "not_measured": 0, "observations": 16, "total": 1.2790559590503108} |
| vendor_wall_seconds | {"count": 16, "not_measured": 0, "observations": 16, "total": 436.28081821100204} |
| native_seconds | {"count": 16, "not_measured": 0, "observations": 16, "total": 395.58467345900135} |
| non_native_overhead_seconds | {"count": 16, "not_measured": 0, "observations": 16, "total": 40.69614475200069} |
| normalization_seconds | {"count": 16, "not_measured": 0, "observations": 16, "total": 0.3006288750329986} |
| state_update_seconds | {"count": 16, "not_measured": 0, "observations": 16, "total": 0.05978679095278494} |
| build_seconds | {"count": 0, "not_measured": 0, "observations": 0, "total": null} |
| failed_native_seconds | {"count": 0, "not_measured": 0, "observations": 0, "total": null} |
| failed_build_seconds | {"count": 0, "not_measured": 0, "observations": 0, "total": null} |

## Interpretation

- Recorded telemetry and hashes do not independently prove legality or causal validity.
- Operational entries are unranked; no paired FPL outcome is assessed.
- An incomplete capture is an observation, not a claim that its process stopped.
- A linked registration is not independently proven to have preceded execution.

## Complete machine record

```json
{
  "experiment": {
    "hypothesis": "Display preserved operational evidence without paired scoring, ranking or promotion.",
    "id": "ownership-control",
    "name": "Ownership control \u2014 completed schedule"
  },
  "limitations": [
    "Recorded telemetry and hashes do not independently prove legality or causal validity.",
    "Operational entries are unranked; no paired FPL outcome is assessed.",
    "An incomplete capture is an observation, not a claim that its process stopped.",
    "A linked registration is not independently proven to have preceded execution."
  ],
  "operations": {
    "assessment": "RECORDED_OPERATIONAL_EVIDENCE_ONLY",
    "completed_origins": [
      137,
      138,
      139,
      140,
      141,
      142,
      143,
      144,
      145,
      146,
      147,
      148,
      149,
      150,
      151,
      152
    ],
    "count_scope": "VENDOR_WRAPPER_INVOCATIONS",
    "counts": {
      "attempted_vendor_calls": 16,
      "failed_vendor_calls": 0,
      "returned_vendor_calls": 16,
      "saved_origins": 16,
      "unknown_in_flight_calls": 0
    },
    "execution_settings": {
      "horizon": 3,
      "max_gw": 152,
      "season": "2023-24",
      "start_gw": 137,
      "time_limit": 120
    },
    "failed_origin": null,
    "failed_stage": null,
    "independent_legality": "NOT ASSESSED",
    "kind": "solio_schedule",
    "performance_assessment": "NOT ASSESSED",
    "promotion_eligible": false,
    "provenance": {
      "basis": "ORIGINAL_CAPTURE_NOT_CURRENT_EXECUTION",
      "runtime": {
        "highspy": "1.15.1",
        "numpy": "2.4.6",
        "pandas": "3.0.3",
        "pulp": "3.3.2",
        "python": "3.11.15",
        "sasoptpy": null,
        "scipy": "1.17.1"
      },
      "source_sha256": {
        "fpl/artifacts.py": "948f427ec7170f644e49cf7d74fc1ef45589079b102c33ffe90cbe0287a813e1",
        "fpl/config.py": "28e01d3614a91156be4a135fc1eb0229dc3f2919ff51226d2827d3d213b70f0b",
        "fpl/milp/_vendor/solio/__init__.py": "868ddfd204cb25ddeff22fbb0454e0836f8345077c3231af5422df8d28e0409e",
        "fpl/milp/_vendor/solio/solver.py": "3026c124f4ee4756fc86fc0447dbdbaf5ebdc512f33591e011e27ceeda5d98c2",
        "fpl/milp/optimize.py": "1c9822442bbf390d9a7b0379cfcfff925ea9ec92e749d4aeda71f38df88a34a5",
        "fpl/milp/origin_prices.py": "9b014344f323e6f7b8c04c38938135f9cdbe531757c9882ce70ee341d4f13b87",
        "fpl/milp/solio.py": "0dc8d125788ad90f45d40fbbb074e03d662a024820a02e988beb6a25b2e8cb21",
        "fpl/milp/solio_inputs.py": "718022707c01c461df4d3a321841a98170772f7ecbe1fef160e6c9c5d99ccc79",
        "fpl/milp/solio_output.py": "24d805fa9288695a9bdf79b2f3817cce534713bae06996033e0be0f49e6b69d9",
        "fpl/milp/solio_receipts.py": "b6a2baacc08661fe130cbeb00b607b450c9e64cb9f2868e5395f52acee58633f"
      },
      "started_sha256": "0ffa03d5e17cd29dfd91be78f1f6c3757348e9ed3550f20589ac16e0c3c035f1",
      "terminal_sha256": "0a75626b35348a734ec68bf88b0459dd90b71e595c44eafe95cf4d8d379a2033"
    },
    "recorded_error": null,
    "requested_origins": [
      137,
      138,
      139,
      140,
      141,
      142,
      143,
      144,
      145,
      146,
      147,
      148,
      149,
      150,
      151,
      152
    ],
    "saved_origins": 16,
    "schema_version": 1,
    "status": "COMPLETED",
    "timings": {
      "admission_seconds": {
        "count": 16,
        "not_measured": 0,
        "observations": 16,
        "total": 2.4300070848839823
      },
      "build_seconds": {
        "count": 0,
        "not_measured": 0,
        "observations": 0,
        "total": null
      },
      "failed_build_seconds": {
        "count": 0,
        "not_measured": 0,
        "observations": 0,
        "total": null
      },
      "failed_native_seconds": {
        "count": 0,
        "not_measured": 0,
        "observations": 0,
        "total": null
      },
      "native_seconds": {
        "count": 16,
        "not_measured": 0,
        "observations": 16,
        "total": 395.58467345900135
      },
      "non_native_overhead_seconds": {
        "count": 16,
        "not_measured": 0,
        "observations": 16,
        "total": 40.69614475200069
      },
      "normalization_seconds": {
        "count": 16,
        "not_measured": 0,
        "observations": 16,
        "total": 0.3006288750329986
      },
      "preparation_seconds": {
        "count": 16,
        "not_measured": 0,
        "observations": 16,
        "total": 1.2790559590503108
      },
      "state_update_seconds": {
        "count": 16,
        "not_measured": 0,
        "observations": 16,
        "total": 0.05978679095278494
      },
      "vendor_wall_seconds": {
        "count": 16,
        "not_measured": 0,
        "observations": 16,
        "total": 436.28081821100204
      }
    }
  },
  "performance": {
    "ci": null,
    "delta": null,
    "points": null,
    "rank": null
  },
  "promotion_eligible": false,
  "registration": {
    "path": "/Users/perandersmolberg/.codex/visualizations/2026/10/17/01a1493e-578c-7060-900d-b428843d8b30/OwnershipCheckpointedPoolReplay_20261018/plan.json",
    "sha256": "80abc0bc04662cebacb5e29faf7be6f94afbd208227bf35f18abd20167578a6d"
  },
  "report_identity": {
    "attempt": "first",
    "implementation": {
      "artifacts.py": "948f427ec7170f644e49cf7d74fc1ef45589079b102c33ffe90cbe0287a813e1",
      "config.py": "28e01d3614a91156be4a135fc1eb0229dc3f2919ff51226d2827d3d213b70f0b",
      "operational_evidence.py": "4ec8b3aedf8d586684bcd4c9a223c04de7672e31633c7a354942bdaf6dab035e",
      "operational_reports.py": "91569abc60a1ddef5431913c92a27b6c6c2d8e3c41127aa10341d2df8f8b52b4"
    },
    "reporting_runtime": {
      "python": "3.11.15"
    },
    "spec_sha256": "4f9b6ccbac3013d9d617bfb636bbb24b02deb9571e7660470f59de451f9bfe2a"
  },
  "report_kind": "OPERATIONAL_EVIDENCE",
  "schema_version": 1
}
```
