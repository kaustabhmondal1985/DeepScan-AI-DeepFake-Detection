import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.config import settings
from app.ml.preprocessing.video_processor import get_video_metadata
from app.ml.preprocessing.frame_extractor import extract_frames
from app.ml.preprocessing.face_detector import detect_faces

from app.ml.analysis_pipeline import DeepScanPipeline


router = APIRouter(
    prefix="/api/analysis",
    tags=["Analysis"],
)


# ============================================================
# Configuration
# ============================================================

API_FRAMES = 16
API_SAMPLE_FPS = 1.0

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# In-memory analysis store
# ============================================================

analyses: Dict[str, Dict[str, Any]] = {}


# ============================================================
# Utility functions
# ============================================================

def generate_analysis_id() -> str:
    """
    Generate a readable DeepScan analysis ID.
    Example: DS-A1B2C3D4
    """
    return f"DS-{uuid.uuid4().hex[:8].upper()}"


def update_progress(
    analysis_id: str,
    progress: int,
    status: str,
) -> None:
    """
    Update analysis progress in memory.
    """
    if analysis_id in analyses:
        analyses[analysis_id]["progress"] = progress
        analyses[analysis_id]["status"] = status


def select_representative_frames(
    frames: list,
    max_frames: int = API_FRAMES,
) -> list:
    """
    Select up to max_frames evenly distributed frames.

    The frame extractor may produce many frames.
    For the API inference pipeline we keep a small,
    consistent number of representative frames.
    """

    if not frames:
        return []

    if len(frames) <= max_frames:
        return frames

    step = (len(frames) - 1) / (max_frames - 1)

    selected = []

    for i in range(max_frames):
        index = round(i * step)
        selected.append(frames[index])

    return selected


def save_json(path: str, data: Dict[str, Any]) -> None:
    """
    Save dictionary as formatted JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Background processing
# ============================================================

def process_video(
    analysis_id: str,
    video_path: str,
    analysis_dir: str,
) -> None:
    """
    Complete DeepScan processing pipeline.

    Stages:

        10%  -> Video metadata
        20%  -> Frame extraction
        30%  -> Representative frame selection
        45%  -> Face detection
        55%  -> Preprocessing data saved
        60%  -> ML inference
        90%  -> Result generation
        100% -> Completed
    """

    try:

        # ----------------------------------------------------
        # Stage 1: Video metadata
        # ----------------------------------------------------

        update_progress(
            analysis_id,
            5,
            "processing",
        )

        metadata = get_video_metadata(video_path)

        analyses[analysis_id]["video"] = metadata

        update_progress(
            analysis_id,
            10,
            "processing",
        )

        print(
            f"[DeepScan] Video metadata extracted: "
            f"{metadata.get('duration_seconds', 0)} seconds"
        )

        # ----------------------------------------------------
        # Stage 2: Frame extraction
        # ----------------------------------------------------

        frames_dir = os.path.join(
            analysis_dir,
            "frames",
        )

        os.makedirs(
            frames_dir,
            exist_ok=True,
        )

        print(
            "[DeepScan] Extracting frames..."
        )

        extraction_result = extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            sample_fps=API_SAMPLE_FPS,
            max_frames=300,
        )

        all_frames = extraction_result.get(
            "frames",
            [],
        )

        print(
            f"[DeepScan] Frames extracted: "
            f"{len(all_frames)}"
        )

        update_progress(
            analysis_id,
            20,
            "processing",
        )

        # ----------------------------------------------------
        # Stage 3: Representative frame selection
        # ----------------------------------------------------

        selected_frames = select_representative_frames(
            all_frames,
            API_FRAMES,
        )

        print(
            f"[DeepScan] Representative frames selected: "
            f"{len(selected_frames)}"
        )

        update_progress(
            analysis_id,
            30,
            "processing",
        )

        # ----------------------------------------------------
        # Stage 4: Face detection
        # ----------------------------------------------------

        print(
            "[DeepScan] Starting face detection..."
        )

        detections = []

        total_selected = len(selected_frames)

        for i, frame_info in enumerate(selected_frames):

            frame_path = frame_info.get("path")

            if not frame_path:
                continue

            try:
                detection = detect_faces(
                    frame_path
                )

                detections.append(
                    {
                        "frame_index": frame_info.get(
                            "index",
                            i,
                        ),
                        "source_frame": frame_info.get(
                            "source_frame",
                            0,
                        ),
                        "timestamp": frame_info.get(
                            "timestamp",
                            0.0,
                        ),
                        "path": frame_path,
                        "faces": detection,
                    }
                )

            except Exception as exc:

                print(
                    f"[DeepScan] Face detection failed "
                    f"for frame {i}: {exc}"
                )

                detections.append(
                    {
                        "frame_index": frame_info.get(
                            "index",
                            i,
                        ),
                        "source_frame": frame_info.get(
                            "source_frame",
                            0,
                        ),
                        "timestamp": frame_info.get(
                            "timestamp",
                            0.0,
                        ),
                        "path": frame_path,
                        "faces": {},
                        "error": str(exc),
                    }
                )

            # Progress from 30 → 45
            if total_selected > 0:

                progress = 30 + int(
                    ((i + 1) / total_selected) * 15
                )

                update_progress(
                    analysis_id,
                    progress,
                    "processing",
                )

        faces_detected = sum(
            len(item.get("faces", {}))
            for item in detections
        )

        print(
            f"[DeepScan] Face detections: "
            f"{faces_detected}"
        )

        # ----------------------------------------------------
        # Stage 5: Save preprocessing information
        # ----------------------------------------------------

        preprocessing = {
            "sample_fps": API_SAMPLE_FPS,
            "frames_available": len(all_frames),
            "frames_analyzed": len(selected_frames),
            "faces_detected": faces_detected,
            "detections": detections,
        }

        preprocessing_path = os.path.join(
            analysis_dir,
            "preprocessing.json",
        )

        save_json(
            preprocessing_path,
            preprocessing,
        )

        analyses[analysis_id][
            "preprocessing"
        ] = preprocessing

        update_progress(
            analysis_id,
            55,
            "processing",
        )

        print(
            "[DeepScan] Preprocessing information saved."
        )

        # ----------------------------------------------------
        # Stage 6: ML inference
        # ----------------------------------------------------

        update_progress(
            analysis_id,
            60,
            "processing",
        )

        print(
            "[DeepScan] Starting ML inference..."
        )

        pipeline = DeepScanPipeline()

        result = pipeline.analyze(
            preprocessing_path
        )

        print(
            "[DeepScan] ML inference completed."
        )

        # ----------------------------------------------------
        # Stage 7: Build final result
        # ----------------------------------------------------

        update_progress(
            analysis_id,
            90,
            "processing",
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "DeepScanPipeline returned invalid result."
            )

        result["analysis_id"] = analysis_id

        result.setdefault(
            "video",
            metadata,
        )

        result.setdefault(
            "preprocessing",
            {
                "frames_analyzed": len(
                    selected_frames
                ),
                "faces_detected": faces_detected,
            },
        )

        result.setdefault(
            "limitations",
            [
                "AI-assisted screening only.",
                "Prediction confidence is model confidence, not proof of manipulation.",
            ],
        )

        # Store result
        analyses[analysis_id]["result"] = result

        # Save result JSON
        result_path = os.path.join(
            analysis_dir,
            "result.json",
        )

        save_json(
            result_path,
            result,
        )

        # ----------------------------------------------------
        # Completed
        # ----------------------------------------------------

        analyses[analysis_id][
            "completed_at"
        ] = datetime.now().isoformat()

        update_progress(
            analysis_id,
            100,
            "completed",
        )

        print(
            f"[DeepScan] Analysis completed: "
            f"{analysis_id}"
        )

    except Exception as exc:

        print(
            f"[DeepScan] Analysis failed: "
            f"{analysis_id}"
        )

        print(
            f"[DeepScan] Error: {exc}"
        )

        analyses[analysis_id]["status"] = "failed"

        analyses[analysis_id]["error"] = str(
            exc
        )

        analyses[analysis_id]["progress"] = 0


# ============================================================
# POST /api/analysis/analyze
# ============================================================

@router.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a video and start DeepScan analysis.
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                f"Allowed formats: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # --------------------------------------------------------
    # Generate analysis ID
    # --------------------------------------------------------

    analysis_id = generate_analysis_id()

    upload_dir = Path(
        settings.UPLOAD_DIR
    )

    analysis_dir = Path(
        settings.OUTPUT_DIR
    ) / analysis_id

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    video_path = upload_dir / (
        f"{analysis_id}{extension}"
    )

    max_bytes = (
        settings.MAX_VIDEO_SIZE_MB
        * 1024
        * 1024
    )

    total_bytes = 0

    try:

        with open(
            video_path,
            "wb",
        ) as buffer:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_bytes:

                    buffer.close()

                    if video_path.exists():
                        video_path.unlink()

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Video exceeds the maximum "
                            f"allowed size of "
                            f"{settings.MAX_VIDEO_SIZE_MB} MB."
                        ),
                    )

                buffer.write(chunk)

    except HTTPException:
        raise

    except Exception as exc:

        if video_path.exists():
            video_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to save uploaded video: "
                f"{exc}"
            ),
        )

    # --------------------------------------------------------
    # Initialize analysis state
    # --------------------------------------------------------

    analyses[analysis_id] = {
        "analysis_id": analysis_id,
        "filename": file.filename,
        "video_path": str(video_path),
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }

    # --------------------------------------------------------
    # Start background processing
    # --------------------------------------------------------

    background_tasks.add_task(
        process_video,
        analysis_id,
        str(video_path),
        str(analysis_dir),
    )

    return {
        "analysis_id": analysis_id,
        "status": "queued",
        "scan_type": "deep",
        "message": (
            "Video uploaded successfully. "
            "Analysis started."
        ),
    }


# ============================================================
# GET /api/analysis/{analysis_id}/status
# ============================================================

@router.get("/{analysis_id}/status")
async def get_analysis_status(
    analysis_id: str,
):
    """
    Get current processing status.
    """

    analysis = analyses.get(
        analysis_id
    )

    if analysis is None:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    return {
        "analysis_id": analysis_id,
        "status": analysis.get(
            "status"
        ),
        "progress": analysis.get(
            "progress",
            0,
        ),
        "error": analysis.get(
            "error"
        ),
    }


# ============================================================
# GET /api/analysis/{analysis_id}/result
# ============================================================

@router.get("/{analysis_id}/result")
async def get_analysis_result(
    analysis_id: str,
):
    """
    Return completed analysis result.
    """

    analysis = analyses.get(
        analysis_id
    )

    if analysis is None:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    if analysis.get("status") == "failed":

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Analysis failed.",
                "error": analysis.get(
                    "error"
                ),
            },
        )

    if analysis.get("status") != "completed":

        return {
            "analysis_id": analysis_id,
            "status": analysis.get(
                "status"
            ),
            "progress": analysis.get(
                "progress",
                0,
            ),
            "message": (
                "Analysis is still processing."
            ),
        }

    return analysis.get(
        "result"
    )