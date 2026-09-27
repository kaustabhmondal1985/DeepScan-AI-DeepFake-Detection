from typing import Dict, Any, List

from app.ml.visual.finetuned_xception import (
    FineTunedXceptionAnalyzer
)


class VisualAnalyzer:

    def __init__(self):

        print(
            "[DeepScan] Initializing "
            "VisualAnalyzer..."
        )

        self.model = (
            FineTunedXceptionAnalyzer()
        )

        print(
            "[DeepScan] VisualAnalyzer ready."
        )


    # ========================================================
    # ANALYZE ONE FACE CROP
    # ========================================================

    def analyze_image(
        self,
        image_path: str
    ) -> Dict[str, Any]:

        return self.model.predict_image(
            image_path
        )


    # ========================================================
    # ANALYZE MULTIPLE FACE CROPS
    # ========================================================

    def analyze_images(
        self,
        image_paths: List[str]
    ) -> Dict[str, Any]:

        return self.model.predict_images(
            image_paths
        )


    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def is_ready(self) -> bool:

        return (
            self.model is not None
        )