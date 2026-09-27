from typing import List, Dict, Any


def calculate_iou(box1, box2):
    """
    Intersection over Union
    """

    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (
        (box1["x2"] - box1["x1"])
        * (box1["y2"] - box1["y1"])
    )

    area2 = (
        (box2["x2"] - box2["x1"])
        * (box2["y2"] - box2["y1"])
    )

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def track_faces(
    frame_results: List[Dict[str, Any]],
    iou_threshold: float = 0.3
):
    """
    Simple IoU-based face tracking.
    """

    tracks = []

    next_track_id = 0

    for frame in frame_results:

        frame_index = frame["frame_index"]

        timestamp = frame["timestamp"]

        faces = frame["faces"]

        for face in faces:

            bbox = face["bbox"]

            best_track = None

            best_iou = 0

            for track in tracks:

                last_detection = track["detections"][-1]

                if (
                    frame_index
                    - last_detection["frame_index"]
                ) > 1:
                    continue

                score = calculate_iou(
                    bbox,
                    last_detection["bbox"]
                )

                if score > best_iou:
                    best_iou = score
                    best_track = track

            if (
                best_track is not None
                and best_iou >= iou_threshold
            ):

                best_track[
                    "detections"
                ].append({

                    "frame_index":
                        frame_index,

                    "timestamp":
                        timestamp,

                    "bbox":
                        bbox,

                    "crop_path":
                        face["crop_path"],

                    "confidence":
                        face["confidence"]
                })

            else:

                tracks.append({

                    "track_id":
                        next_track_id,

                    "detections": [

                        {
                            "frame_index":
                                frame_index,

                            "timestamp":
                                timestamp,

                            "bbox":
                                bbox,

                            "crop_path":
                                face["crop_path"],

                            "confidence":
                                face["confidence"]
                        }

                    ]
                })

                next_track_id += 1

    return {
        "total_tracks": len(tracks),
        "tracks": tracks
    }