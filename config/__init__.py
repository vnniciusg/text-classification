from typing import Optional

from pydantic_settings import BaseSettings


class Config(BaseSettings):

    ENV: str = "dev"
    BUFFER_SIZE: int = 10000
    BATCH_SIZE: int = 64
    VOCAB_SIZE: int = 1000
    OUTPUT_SEQUENCE_LENGTH: Optional[int] = None
    TRAINING_LEARNING_RATE: float = 1e-3
    EPOCHS: int = 10
