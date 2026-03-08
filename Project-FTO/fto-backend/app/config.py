from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    allowed_email_domains: str = "pmintl.net,iteso.mx"  # Comma-separated domains
    frontend_url: str = "http://localhost:5173"
    environment: str = "development"

    @property
    def email_domains_list(self) -> list[str]:
        return [d.strip().lower() for d in self.allowed_email_domains.split(",")]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
