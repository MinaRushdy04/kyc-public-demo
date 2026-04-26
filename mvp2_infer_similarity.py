import argparse
import json

from modules.face_similarity import ArcFaceOnnxEmbedder, MobileNetFaceBaseline, score_pair
from modules.utils import simple_yaml_load


def decision(score: float, low: float, high: float) -> tuple[str, str]:
    if score >= high:
        return "FACE_MATCH_ACCEPTED", "ACCEPT_FOR_NEXT_STEP"
    if score >= low:
        return "FACE_MATCH_MANUAL_REVIEW", "MANUAL_REVIEW"
    return "FACE_MATCH_REJECTED", "REJECT_RETRY"


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP2 ID-face to selfie-face similarity inference.")
    parser.add_argument("--config", default="config_mvp2.yaml")
    parser.add_argument("--id-image", required=True)
    parser.add_argument("--selfie-image", required=True)
    args = parser.parse_args()

    cfg = simple_yaml_load(args.config)
    if str(cfg.get("face_match_model", "")).lower() == "arcface_onnx":
        embedder = ArcFaceOnnxEmbedder(str(cfg["arcface_onnx_path"]))
    else:
        embedder = MobileNetFaceBaseline(image_size=int(cfg.get("image_size", 224)))
    result = score_pair(args.id_image, args.selfie_image, embedder)
    status, next_decision = decision(
        result["similarity"],
        low=float(cfg.get("manual_review_low", 0.50)),
        high=float(cfg.get("manual_review_high", 0.70)),
    )
    print(
        json.dumps(
            {
                "status": status,
                "decision": next_decision,
                "similarity": round(float(result["similarity"]), 6),
                "manual_review_low": float(cfg.get("manual_review_low", 0.50)),
                "auto_accept_threshold": float(cfg.get("manual_review_high", 0.70)),
                "id_face_found": bool(result["id_face_found"]),
                "selfie_face_found": bool(result["selfie_face_found"]),
                "mvp_version": "MVP2",
                "warning": "MVP2 compares face similarity only. It does not perform liveness or document authenticity checks.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
