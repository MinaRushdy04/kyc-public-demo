# MVP2 Face Similarity Baseline Report

- Dataset: `ud-biometrics/Selfie-and-ID-Dataset`
- Dataset URL: https://huggingface.co/datasets/ud-biometrics/Selfie-and-ID-Dataset
- License: `cc-by-nc-nd-4.0`
- Baseline model: `arcface_onnx`
- Valid pairs scored: **472**
- Same-person pairs: **236**
- Different-person pairs: **236**
- ROC-AUC: **0.8001831370295892**
- ID face detection misses: **48**
- Selfie face detection misses: **32**

## Threshold 0.70

- threshold: **0.7**
- accuracy: **0.586864406779661**
- precision: **0.8059701492537313**
- recall: **0.2288135593220339**
- f1: **0.3564356435643564**
- false_accepts: **13**
- false_rejects: **182**
- far: **0.05508474576271186**
- frr: **0.7711864406779662**

## Threshold Targeting ~1% FAR On This Tiny Sample

- threshold: **0.9632531867752075**
- accuracy: **0.5466101694915254**
- precision: **0.9230769230769231**
- recall: **0.1016949152542373**
- f1: **0.183206106870229**
- false_accepts: **2**
- false_rejects: **212**
- far: **0.00847457627118644**
- frr: **0.8983050847457628**

## Verdict

- This is a baseline MVP2 harness, not a production biometric model.
- For production, replace this ImageNet embedding baseline with ArcFace/InsightFace or an equivalent face-recognition embedding model, then calibrate on consented KYC-style Egyptian/region-relevant samples.
- Liveness/anti-spoofing remains out of scope per current backend constraints.
