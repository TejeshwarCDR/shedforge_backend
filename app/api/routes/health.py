from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import inspect, text

from app.core.config import get_settings
from app.db.session import engine

router = APIRouter()

settings = get_settings()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/live")
def health_live() -> dict:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready")
def health_ready() -> JSONResponse:
    required_columns = {
        "users": {
            "id",
            "email",
            "role",
            "section_name",
            "program_id",
            "semester_number",
            "batch_year",
            "roll_number",
        },
        "faculty": {"id", "email", "program_id", "preferred_subject_codes", "semester_preferences"},
        "courses": {
            "id",
            "code",
            "program_id",
            "semester_number",
            "batch_year",
            "theory_hours",
            "lab_hours",
            "tutorial_hours",
            "batch_segregation",
            "practical_contiguous_slots",
            "assign_faculty",
            "assign_classroom",
            "default_room_id",
            "elective_category",
        },
        "institution_settings": {"id", "academic_year", "semester_cycle"},
    }
    db_ok = True
    missing_tables: list[str] = []
    missing_columns: dict[str, list[str]] = {}
    db_error: str | None = None

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            inspector = inspect(connection)
            table_names = set(inspector.get_table_names())
            for table_name, columns in required_columns.items():
                if table_name not in table_names:
                    missing_tables.append(table_name)
                    continue
                existing = {item["name"] for item in inspector.get_columns(table_name)}
                missing = sorted(columns - existing)
                if missing:
                    missing_columns[table_name] = missing
    except Exception as exc:  # pragma: no cover - environment dependent
        db_ok = False
        db_error = str(exc)

    schema_ok = not missing_tables and not missing_columns
    ready = db_ok and schema_ok
    smtp_configured = bool(settings.smtp_host and settings.smtp_from_email)

    payload = {
        "status": "ok" if ready else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {
            "ok": db_ok,
            "schema_ok": schema_ok,
            "missing_tables": missing_tables,
            "missing_columns": missing_columns,
            "error": db_error,
        },
        "smtp": {
            "configured": smtp_configured,
            "host": settings.smtp_host,
            "port": settings.smtp_port,
            "from_email": settings.smtp_from_email,
            "use_tls": settings.smtp_use_tls,
            "use_ssl": settings.smtp_use_ssl,
        },
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)
