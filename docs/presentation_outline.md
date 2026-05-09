# Presentation Outline

## Slide 1 — Title
- Project: KYC Validator System for Egyptian National ID Verification
- Team/student info
- Context: ride-hailing onboarding

## Slide 2 — Problem and Motivation
**Talking points**
- Why onboarding fraud matters
- Why automated pre-screening is useful
- Why false accepts are high risk

## Slide 3 — Scope Boundaries
**Show clearly**
- Implemented now (MVP1 + MVP2)
- Deferred (OCR, DB verification, liveness, full anti-spoofing, monitoring)

## Slide 4 — Architecture Overview
**Suggested visual**
- Two-stage flow: document validation then face similarity
- Decision outputs: reject/review/accept-for-next-step

## Slide 5 — MVP1 (Document Validation)
**Talking points**
- Binary classification task
- EfficientNet transfer learning approach
- Threshold-based gate
- Why hard negatives were added

**Suggested screenshot**
- MVP1 Streamlit prediction result screen

## Slide 6 — Dataset Engineering for MVP1
**Talking points**
- Positive/negative construction
- FantasyID extraction and selection process
- Dataset audit and readiness checks

## Slide 7 — MVP2 (Face Similarity)
**Talking points**
- Face detection, alignment, embeddings, cosine scoring
- Quality handling path (blur/small-face issues)
- Prototype nature of current calibration

**Suggested screenshot**
- Face app output with score + decision

## Slide 8 — Calibration and Threshold Tradeoffs
**Talking points**
- FAR vs FRR tradeoff
- Why stricter threshold reduces FAR but increases FRR
- Internal testing interpretation only

## Slide 9 — Prototype Results Snapshot
**Use real values only**
- Example: ROC-AUC and threshold behavior from reports
- Emphasize dataset limitations

## Slide 10 — Risks and Limitations
**Talking points**
- Fake IDs, replay attacks, no liveness
- Small calibration datasets
- Not production-hardened yet

## Slide 11 — Future Work Roadmap
**Priority order**
1. Better data + calibration
2. Liveness/anti-spoofing
3. OCR + credential matching
4. Monitoring + versioning + backend integration

## Slide 12 — Conclusion
**Talking points**
- What was successfully implemented
- Why this is a solid prototype baseline
- What must happen before production use

---

## Suggested Demo Order (Short Live Demo)
1. Run MVP1 app on valid vs invalid document examples.
2. Run MVP2 two-image face match example.
3. Run single-image selfie-holding-ID flow (if stable sample available).
4. Show one failure/edge case and explain fallback decision.

## Suggested Screenshots
- MVP1 class probability + decision output
- MVP2 similarity score + decision bands
- Single-image flow with detected holder face and ID crop
- Example report snippet with threshold tradeoff metrics

## Possible Viva Questions (with concise answers)

1. **Why separate MVP1 and MVP2 instead of one end-to-end model?**  
   Separation simplifies debugging, allows independent thresholding, and supports phased deployment risk control.

2. **Why is false accept treated as more dangerous?**  
   Because it allows potentially invalid identities to continue onboarding, increasing fraud exposure.

3. **Why not claim production readiness now?**  
   Missing liveness, spoof resistance, large-scale calibration, and operational monitoring.

4. **Why use pretrained models instead of training from scratch?**  
   Faster prototyping, lower compute cost, and practical baseline performance for system integration.

5. **How do you justify threshold values?**  
   Through calibration reports showing FAR/FRR tradeoffs on available paired data; values remain provisional pending larger real data.

6. **What is the biggest technical risk today?**  
   Attack resistance (replay/spoof) and threshold instability under real-world distribution shifts.

## Risk Questions Examiners May Ask

- **Can a printed photo bypass MVP2?**  
  Potentially yes; liveness/anti-spoofing is not yet implemented.

- **Can high-quality fake IDs bypass MVP1?**  
  Possible, especially if fake style resembles seen training distributions.

- **Are your metrics enough for deployment?**  
  No. Current metrics are useful for internal validation, not production sign-off.

- **What prevents model drift in production?**  
  Currently no full monitoring stack; this is a planned future-work area.g