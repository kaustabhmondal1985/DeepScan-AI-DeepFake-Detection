from pprint import pprint

from app.ml.preprocessing.face_tracker import (
    track_faces
)

sample_frames = [

    {
        "frame_index": 0,
        "timestamp": 0,
        "faces": [
            {
                "bbox": {
                    "x1": 100,
                    "y1": 100,
                    "x2": 200,
                    "y2": 200
                },
                "crop_path": "a.jpg",
                "confidence": 0.99
            }
        ]
    },

    {
        "frame_index": 1,
        "timestamp": 1,
        "faces": [
            {
                "bbox": {
                    "x1": 105,
                    "y1": 105,
                    "x2": 205,
                    "y2": 205
                },
                "crop_path": "b.jpg",
                "confidence": 0.98
            }
        ]
    }
]

result = track_faces(
    sample_frames
)

pprint(result)