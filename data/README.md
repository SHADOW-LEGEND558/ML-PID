# Data access

Primary data: `pid_dataset_final_10000.csv` and its `.meta.json` sidecar, included directly. It contains 10,000 accepted synthetic plants from 10,083 attempts, including 1,027 exact-zero-delay plants. Rejected attempts are in `generation_final_10000.log`.

Each row stores physical audit metadata, five empirical inputs, three numerical PID gains and recording duration. Splits/seeds are in `results/dataset_final_10000/protocol.json`. Original CRLF-file SHA-256: `e86f0039c0747afa186a322bdc89f77b99f03a1ba9aaa77a40c6243e64d97089`. The equivalent LF checkout has SHA-256 `e20ca92bc46cb95539a09271c58e8d1649a8b2cad3712ae74f21fbb93a2c4d29`. Integrity checks accept either line-ending representation; changing data values still fails verification.

The 20K and 1K CSVs are development cohorts, not the final test population. The 20K data supplies independent training plants for the extended learning curve and supports overlap checks. The 1K data supports historical method comparisons. The obsolete unregularized CSV, old generation logs and finalization console log were removed, along with the 20K rejected-plant experiment. The final generation log remains necessary for the final 83-rejected-plant analysis. No dataset contains private measurements or personal information.
