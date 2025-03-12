from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "HairVision"

    class Config:
        env_file = ".env"


settings = Settings()
