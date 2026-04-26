# MVP2 Face Matching

MVP2 compares the face photo visible on an Egyptian National ID with the user selfie and returns a similarity score plus a 3-way decision.

This module does not build backend code, MongoDB code, OCR extraction, OCR/database matching, liveness, anti-spoofing, or new face model training.

## Scope

- Detect face in the ID image using SCRFD-style ONNX face detection.
- Detect face in the selfie image.
- Reject/warn when the selfie has no face or multiple large faces.
- Align both faces using 5 facial landmarks.
- Generate ArcFace embeddings with a pretrained ONNX model.
- Compute cosine similarity and convert it to a score from 0 to 100.
- Return `AUTO_REJECT`, `MANUAL_REVIEW`, or `AUTO_ACCEPT`.

## Limitations

- No liveness or anti-spoofing is included.
- This compares face similarity only.
- Printed-photo and screen-replay attacks are not reliably prevented.
- This does not prove full KYC identity alone.
- This does not do OCR or database matching.
- Production calibration requires more real paired ID/selfie data collected with consent.

## Thresholds

Configured in `config_face.yaml`:

- `score < 50`: `AUTO_REJECT`, `FACE_MISMATCH`
- `50 <= score < 70`: `MANUAL_REVIEW`, `FACE_UNCERTAIN`
- `score >= 70`: `AUTO_ACCEPT`, `FACE_MATCH`

After calibration, runtime can use profile-specific thresholds from:

```text
mvp2_face_match/reports/face_score_calibration_stable.json
mvp2_face_match/reports/face_score_calibration_accurate.json
```

The current default runtime mode is `raw_score_with_calibrated_thresholds`: cosine is still the raw embedding metric, but the accept/reject thresholds are selected from FAR/FRR evaluation instead of hardcoded `50/70`.

## Models

- Detector: `scrfd_10g_bnkps.onnx`
- Stable embedder: `w600k_mbf.onnx`
- Accurate embedder: `arc.onnx`

Both are pretrained inference models. MVP2 does not train from scratch.
The Streamlit single-image app can switch between `accurate` and `stable` profiles. The accurate profile gives stronger embeddings but uses more RAM, so it runs only inside an isolated subprocess.

## Calibration Dataset

MVP2 does not train a new face recognition model. It uses pretrained SCRFD/ArcFace-style inference and needs paired data for threshold calibration.

Expected calibration layout:

```text
mvp2_face_match/calibration_dataset/
  identities/
    person_001/
      id/
      selfies/
    person_002/
      id/
      selfies/
```

Build calibration pairs:

```powershell
python mvp2_face_match/build_face_pairs_dataset.py
```

Use the old tiny public sample only for prototype smoke testing:

```powershell
python mvp2_face_match/build_face_pairs_dataset.py --legacy-sample
```

Evaluate/calibrate thresholds:

```powershell
python mvp2_face_match/evaluate_face_pairs.py --pairs mvp2_face_match/reports/face_pairs.csv --face-profile stable --write-calibration
python mvp2_face_match/evaluate_face_pairs.py --pairs mvp2_face_match/reports/face_pairs.csv --face-profile accurate --write-calibration
```

The evaluator reports score distributions, FAR/FPR, FRR/FNR, accuracy at thresholds `50`, `60`, `70`, `80`, and a recommended threshold. Failed detections and low-quality faces are excluded from score calibration and reported separately.

## App Modes

There are now three demo modes in the project:

1. MVP1 document validator:
   one image -> document validator only.

2. MVP2 two-upload face matcher:
   ID image + selfie image -> face match only.

3. MVP2 single-image selfie-holding-ID:
   one full image -> locate ID, validate document with MVP1, then match holder face against ID portrait.

The single-image app is the least confusing UX for a selfie-holding-ID flow, but ID localization is still MVP-level.

## Single-Pair Face Inference

```powershell
cd D:\Projects\KYC\kyc-validator-training
python mvp2_face_match/face_match.py --id-image <path> --selfie-image <path>
```

## Streamlit

```powershell
cd D:\Projects\KYC\kyc-validator-training
streamlit run mvp2_face_match/app_streamlit_face.py
```

## Single-Image Selfie-Holding-ID App

```powershell
cd D:\Projects\KYC\kyc-validator-training
streamlit run mvp2_face_match/app_streamlit_single_image.py
```

CLI:

```powershell
python mvp2_face_match/full_kyc_single_image.py --image <path>
```

This pipeline:

1. Locates a card-like crop.
2. Scores candidate crops with the MVP1 document validator.
3. Rejects if no candidate passes MVP1.
4. Detects the holder face from the original full image, preferring a face outside the ID-card bounding box.
5. Enhances/upscales the cropped ID card only for ID-portrait detection when needed.
6. Detects the ID portrait face from the cropped or enhanced ID.
7. Applies quality gates for blur, brightness, glare, face scale, and landmark pose stability.
8. Aligns both faces and compares ArcFace embeddings.
9. Returns `ACCEPT_FOR_NEXT_STEP`, `MANUAL_REVIEW`, or `REJECT_RETRY`.

The Streamlit app also accepts up to three still images from the same attempt and selects the strongest frame by document status, face status, and score. This is a practical still-image alternative to video/liveness when the backend cannot stream video.

Important: the result means “KYC step accepted for next stage,” not “identity verified.”

The Streamlit app can save hard cases into:

```text
mvp2_face_match/outputs/hard_pairs/
```

Hard cases are not automatically added to training. They are for review and future dataset curation.

## Evaluation

Build a prototype pair CSV:

```powershell
python mvp2_face_match/build_face_pairs_dataset.py
```

Evaluate pairs:

```powershell
python mvp2_face_match/evaluate_face_pairs.py --pairs mvp2_face_match/reports/face_pairs.csv
```

The included public selfie/ID dataset is suitable for prototype testing only, not production calibration.
