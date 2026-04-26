from __future__ import annotations


LIMITATIONS = [
    "No liveness or anti-spoofing is included.",
    "This compares face similarity only.",
    "Printed-photo and screen-replay attacks are not reliably prevented.",
]


def response(
    status: str,
    decision: str,
    score: float | None,
    cosine_similarity: float | None,
    cfg: dict,
    id_face_detected: bool,
    selfie_face_detected: bool,
    selfie_face_count: int,
    id_quality_ok: bool,
    selfie_quality_ok: bool,
    warnings: list[str],
) -> dict:
    return {
        "status": status,
        "decision": decision,
        "score": None if score is None else round(float(score), 4),
        "cosine_similarity": None if cosine_similarity is None else round(float(cosine_similarity), 6),
        "thresholds": {
            "auto_reject_below": float(cfg.get("auto_reject_below", 50)),
            "auto_accept_at_or_above": float(cfg.get("auto_accept_at_or_above", 70)),
        },
        "faces": {
            "id_face_detected": bool(id_face_detected),
            "selfie_face_detected": bool(selfie_face_detected),
            "selfie_face_count": int(selfie_face_count),
        },
        "quality": {
            "id_face_quality_ok": bool(id_quality_ok),
            "selfie_face_quality_ok": bool(selfie_quality_ok),
            "warnings": warnings,
        },
        "mvp_version": "MVP2",
        "limitations": LIMITATIONS,
    }
