import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = "DeepScan API"
    APP_VERSION: str = "1.0.0"

    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    MAX_VIDEO_SIZE_MB: int = int(
        os.getenv("MAX_VIDEO_SIZE_MB", "500")
    )

    UPLOAD_DIR: str = os.getenv(
        "UPLOAD_DIR",
        "uploads"
    )

    OUTPUT_DIR: str = os.getenv(
        "OUTPUT_DIR",
        "outputs"
    )


settings = Settings()