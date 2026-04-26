# KYC Validator Training (MVP1)

This project prepares data and training scripts for **MVP1 document validation only**:

- Task: `national_id` vs `not_national_id`
- Scope: Egyptian **front** National ID detection/classification at document level
- Decision threshold target: `0.95` for `national_id_probability`

## MVP1 Scope Boundaries

MVP1 does **document validation only**.

- Included:
  - Binary classifier for `national_id` vs `not_national_id`
  - Dataset preparation and audit
- Not included:
  - User identity verification
  - OCR matching
  - Credential/database matching
  - Face/image matching
  - Manual review workflow

Backend business logic remains the backend team's responsibility.

## Data Separation

- Raw read-only data: `D:\Datasets`
- Project code + prepared data + reports + model artifacts:
  - `D:\Projects\KYC\kyc-validator-training`

The pipeline does not modify or delete original files under `D:\Datasets`.

## Project Structure

```text
D:\Projects\KYC\kyc-validator-training
  README.md
  config.yaml
  requirements.txt
  extract_fantasyid.py
  audit_fantasyid.py
  select_fantasyid_negatives.py
  prepare_validator_dataset.py
  audit_prepared_dataset.py
  train_validator.py
  evaluate_validator.py
  infer_validator.py
  export_onnx.py
  modules\
  reports\
  models\
  outputs\
    sample_grids\
    false_accepts\
    false_rejects\
    debug\
```

## Run Order

Run from `D:\Projects\KYC\kyc-validator-training`:

```powershell
python extract_fantasyid.py --dataset-root D:\Datasets --extract-dir D:\Datasets\FantasyID_extracted
python audit_fantasyid.py --extract-dir D:\Datasets\FantasyID_extracted --report-path reports\fantasyid_audit.md
python select_fantasyid_negatives.py --extract-dir D:\Datasets\FantasyID_extracted --output-dir prepared_validator_dataset\not_national_id\fantasyid_selected --report-path reports\fantasyid_selection_report.md
python prepare_validator_dataset.py --dataset-root D:\Datasets --national-out prepared_validator_dataset\national_id --mapping-csv reports\source_mapping.csv
python audit_prepared_dataset.py --prepared-root prepared_validator_dataset --report-path reports\prepared_validator_audit.md --distribution-csv reports\class_distribution.csv
```

## Report Interpretation

- `reports/fantasyid_audit.md`: extraction validity and FantasyID content profile
- `reports/fantasyid_selection_report.md`: negative selection counts and sufficiency
- `reports/source_mapping.csv`: exact origin of each positive crop
- `reports/prepared_validator_audit.md`: final class quality, duplicates, leakage flags, and `READY_TO_TRAIN`
- `reports/class_distribution.csv`: final class counts

## Training (Only When Data Is Ready)

Train only after `READY_TO_TRAIN: yes`:

```powershell
python train_validator.py --config config.yaml
python evaluate_validator.py --config config.yaml
python infer_validator.py --config config.yaml --image <path_to_image>
python export_onnx.py --config config.yaml
python -m streamlit run app_streamlit.py
```

## Why False Accepts Are Dangerous

A false accept means a non-Egyptian or invalid document is predicted as `national_id` and passes to the next step. In KYC, this is higher risk than false reject because it can enable fraud attempts to enter downstream verification stages. For MVP1, keep strong focus on minimizing false accepts and review `outputs/false_accepts` after evaluation.
