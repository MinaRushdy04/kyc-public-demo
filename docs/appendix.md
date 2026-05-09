\# Appendix



\## A. Threshold Terminology

A threshold is a cutoff used to convert a continuous model score into a decision. In this project, thresholds determine whether to auto-reject, manual-review, or auto-accept.



\## B. FAR/FPR vs FRR/FNR

\- \*\*FAR / FPR (False Accept Rate / False Positive Rate):\*\* invalid case accepted.

\- \*\*FRR / FNR (False Reject Rate / False Negative Rate):\*\* valid case rejected.



For KYC risk, FAR is often prioritized because false accepts can enable fraud progression.



\## C. Cosine Similarity (Face Matching)

Cosine similarity measures directional closeness between two embedding vectors. Higher similarity usually means higher likelihood of same person, but it is not identity proof by itself.



\## D. Why ArcFace-Style Embeddings

ArcFace/InsightFace-style embeddings are a practical off-the-shelf choice for prototype face similarity due to strong ecosystem support, inference availability in ONNX, and stable behavior for threshold calibration workflows.



\## E. Why EfficientNet Transfer Learning

EfficientNet transfer learning offers a practical balance:

\- good baseline performance for image classification,

\- manageable compute requirements,

\- faster iteration than full training from scratch.



\## F. Hard-Negative Mining

Hard negatives are non-target images that look similar to valid IDs. Adding them during dataset preparation helps the document classifier learn stronger boundaries and reduces risky false accepts.



\## G. Why Streamlit Testing Apps

Streamlit apps provide a quick internal interface to:

\- visualize predictions,

\- inspect edge cases,

\- test threshold impact,

\- support demo and operator-oriented discussion.



\## H. Calibration Dataset Structure

MVP2 calibration expects identity-wise grouping (ID images + selfies per person) to build same-person and different-person pairs for threshold analysis.



\## I. False Accepts vs False Rejects

\- \*\*False accept:\*\* higher fraud risk.

\- \*\*False reject:\*\* higher user friction and operational review load.



Design decisions in this project intentionally bias toward reducing dangerous false accepts in early KYC gating.



\## J. Glossary

\- \*\*KYC:\*\* Know Your Customer.

\- \*\*MVP:\*\* Minimum Viable Product.

\- \*\*Embedding:\*\* numeric vector representation of an image/face.

\- \*\*Calibration:\*\* selecting operating thresholds based on evaluation tradeoffs.

\- \*\*Manual Review:\*\* human verification fallback state.

\- \*\*Replay Attack:\*\* attack using photo/screen/video rather than live person.

\- \*\*Hard Case:\*\* difficult sample kept for analysis and future curation.



