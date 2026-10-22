# 🔬 Verification notes

Run `python Code/run_all.py` from the repository root. The runner stops at the first failed script and saves each script's output under `Reports/logs/`.

| Script | Scope |
|---|---|
| s01 | Scalar minimax optimization checks |
| s02 | Independent identities and finite-horizon checks |
| s03 | Max-plus recurrence checks |
| s04 | Exact computations on selected time scales |
| s05 | Numerical experiments |
| s06 | Sampling-design calculations |
| s07 | Network experiments |
| s12 | Boundary-case checks |
| s13 | Tail-disturbance counterexample check |

The runner intentionally excludes manuscript, table and report compilation. Numerical tests and exact-arithmetic checks complement analytical proofs; a successful run is not a formal proof of every theorem or a claim of publication novelty.

This repository package includes project source and original project visuals. It contains no manuscript LaTeX, PDF, bundled CSV datasets, reference books, third-party source code or vendor libraries. Dependencies are installed separately. Scripts may generate CSV outputs locally during reproduction; these are ignored by Git.

No open-source license has been selected in this package. The author can choose licensing terms before public release.

## Package validation

All nine scripts passed during preparation on 22 October 2026. The four GIFs were generated and visually checked. Archive contents were checked to exclude LaTeX, PDF and CSV files.
