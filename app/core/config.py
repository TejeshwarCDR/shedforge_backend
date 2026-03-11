from functools import lru_cache
import json
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # Resolve to backend/.env so `uvicorn --app-dir backend` works from any cwd.
    model_config = SettingsConfigDict(env_file=str(BACKEND_ENV_FILE), env_file_encoding="utf-8")

    project_name: str = "ShedForge API"
    api_prefix: str = "/api"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/shedforge"
    skip_runtime_schema_bootstrap: bool = False

    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    auth_enabled: bool = True
    reset_token_expire_minutes: int = 30
    expose_reset_token: bool = True
    password_reset_enabled: bool = False
    login_otp_enabled: bool = False
    login_otp_expire_minutes: int = 10
    login_otp_max_attempts: int = 5
    expose_login_otp: bool = False
    login_otp_log_to_terminal: bool = False
    login_otp_allow_terminal_fallback: bool = False
    auth_rate_limit_window_seconds: int = 300
    auth_rate_limit_register_max_requests: int = 8
    auth_rate_limit_login_max_requests: int = 12
    auth_rate_limit_otp_request_max_requests: int = 8
    auth_rate_limit_otp_verify_max_requests: int = 15
    auth_rate_limit_password_reset_max_requests: int = 8

    smtp_host: str | None = None
    email_enabled: bool = False
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from_email: str | None = None
    smtp_from_name: str = "ShedForge"
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_backup_host: str | None = None
    smtp_backup_port: int = 587
    smtp_backup_username: str | None = None
    smtp_backup_password: str | None = None
    smtp_backup_from_email: str | None = None
    smtp_backup_from_name: str = "ShedForge Backup"
    smtp_backup_use_tls: bool = True
    smtp_backup_use_ssl: bool = False
    smtp_notification_prefer_backup: bool = False
    smtp_retry_attempts: int = 2
    smtp_retry_backoff_seconds: float = 1.0
    smtp_rate_limit_cooldown_seconds: int = 1800
    smtp_timeout_seconds: int = 15

    sendgrid_api_key: str | None = None

    max_request_size_bytes: int = 2_500_000
    security_enable_hsts: bool = False
    security_hsts_max_age_seconds: int = 31536000

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
    cors_origin_regex: str | None = (
        r"^https?://("
        r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
        r")(:\d+)?$"
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
