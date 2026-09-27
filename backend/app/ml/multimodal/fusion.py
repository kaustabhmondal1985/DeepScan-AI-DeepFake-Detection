from typing import Dict, Any


def fuse_multimodal_evidence(
    visual_result: Dict[str, Any],
    audio_result: Dict[str, Any],
    av_result: Dict[str, Any]
) -> Dict[str, Any]:

    visual_available = (
        visual_result.get("status") != "unavailable"
    )

    audio_available = (
        audio_result.get("status") != "unavailable"
    )

    av_available = (
        av_result.get("status") == "success"
    )

    evidence_sources = []

    if visual_available:
        evidence_sources.append("visual")

    if audio_available:
        evidence_sources.append("audio")

    if av_available:
        evidence_sources.append("audio_visual")

    # We deliberately do NOT calculate a fake
    # deepfake probability at this stage.
    assessment = "Inconclusive"

    return {
        "assessment": assessment,
        "confidence": None,

        "evidence_sources": evidence_sources,

        "visual": visual_result,
        "audio": audio_result,
        "audio_visual": av_result,

        "message": (
            "Multimodal evidence collected successfully. "
            "A trained fusion classifier is required "
            "before producing a deepfake probability."
        )
    }