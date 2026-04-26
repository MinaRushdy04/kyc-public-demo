# MVP2 Face Pair Calibration Evaluation

**Suitable for prototype testing and threshold calibration only. Do not train ArcFace from this report.**

- Total pairs: **472**
- Scorable pairs used for calibration: **396**
- Excluded failed/low-quality pairs: **76**
- Positive scorable pairs: **198**
- Negative scorable pairs: **198**
- ROC-AUC on scorable pairs: **0.9997959391898785**

## Fixed Thresholds

| threshold | accuracy | FAR/FPR | FRR/FNR | precision | recall | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.6995 | 0.6010 | 0.0000 | 0.6246 | 1.0000 | 198 | 119 | 79 | 0 |
| 60 | 0.9924 | 0.0000 | 0.0152 | 1.0000 | 0.9848 | 195 | 0 | 198 | 3 |
| 70 | 0.9141 | 0.0000 | 0.1717 | 1.0000 | 0.8283 | 164 | 0 | 198 | 34 |
| 80 | 0.6111 | 0.0000 | 0.7778 | 1.0000 | 0.2222 | 44 | 0 | 198 | 154 |

## Calibrated Match Probability

- Model profile: **stable**
- Calibration artifact: **D:\Projects\KYC\kyc-validator-training\mvp2_face_match\reports\face_score_calibration_stable.json**
- Score mode after calibration: **calibrated_match_probability**
- Auto-reject below: **36.0**
- Auto-accept at or above: **36.0**

| calibrated threshold | accuracy | FAR/FPR | FRR/FNR | precision | recall | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.9848 | 0.0000 | 0.0303 | 1.0000 | 0.9697 | 192 | 0 | 198 | 6 |
| 60 | 0.9571 | 0.0000 | 0.0859 | 1.0000 | 0.9141 | 181 | 0 | 198 | 17 |
| 70 | 0.9343 | 0.0000 | 0.1313 | 1.0000 | 0.8687 | 172 | 0 | 198 | 26 |
| 80 | 0.9167 | 0.0000 | 0.1667 | 1.0000 | 0.8333 | 165 | 0 | 198 | 33 |

## Recommended Threshold

- Target: highest recall while keeping FAR/FPR <= 2% on this calibration set.
- Recommendation: **60.0**
- FAR/FPR: **0.0**
- FRR/FNR: **0.015151515151515152**
- Accuracy: **0.9924242424242424**

## Quality Handling

- Failed detections and low-quality faces are excluded from threshold math.
- `LOW_QUALITY_ID_FACE` and `LOW_QUALITY_SELFIE_FACE` should trigger `REJECT_RETRY`, not silent zero-similarity scoring.

## Production Warning

- No liveness or anti-spoofing is included.
- Printed-photo and screen-replay attacks are not reliably prevented.
- Production calibration requires consented, representative ID/selfie pairs from the real capture flow.
