# Prepared Validator Dataset Audit

- Prepared root: `D:\Projects\KYC\kyc-validator-training\prepared_validator_dataset`
- Audit time (UTC): `2026-04-25T19:26:04.608759+00:00`

## Counts
- national_id count: **956**
- not_national_id count: **2274**
- class imbalance ratio (max/min): **2.38x**

## Quality
- Corrupted/unreadable images: **0**
- Exact duplicate groups: **0**
- Exact duplicate file count: **0**
- Near duplicate pairs (dHash <= 6): **493613**
- Cross-class exact duplicates: **0**
- Cross-class near duplicates: **0**
- Width summary: **min=100, median=929, max=3242**
- Height summary: **min=41, median=580, max=2013**

## Leakage and Contamination Checks
- Possible leakage (cross-class exact/near): **0**
- Filename-based back_egy_id contamination hints: **0**
- Source-mapping back_egy_id rows: **0**

## Sample Grids
- national_id grid: `D:\Projects\KYC\kyc-validator-training\outputs\sample_grids\national_id_grid.png`
- not_national_id grid: `D:\Projects\KYC\kyc-validator-training\outputs\sample_grids\not_national_id_grid.png`

## Final Verdict
- READY_TO_TRAIN: **yes**
