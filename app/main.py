from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    activity,
    auth,
    conflicts,
    constraints,
    courses,
    feedback,
    faculty,
    generator,
    health,
    issues,
    leaves,
    notifications,
    programs,
    program_structure,
    rooms,
    settings as settings_routes,
    students,
    system,
    timetable,
)
from app.core.config import get_settings
from app.core.middleware import RequestSizeLimitMiddleware, SecurityHeadersMiddleware
from app.db.bootstrap import ensure_runtime_schema_compatibility

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    import logging

    logger = logging.getLogger("uvicorn.error")
    if settings.skip_runtime_schema_bootstrap:
        logger.warning("Skipping runtime schema compatibility bootstrap because SKIP_RUNTIME_SCHEMA_BOOTSTRAP=true")
    else:
        try:
            ensure_runtime_schema_compatibility()
        except Exception as e:
            logger.error(f"\n🚨 DATABASE CONNECTION FAILED: {e}")
            logger.error("The application started, but many features will fail until Supabase is reachable.\n")
    yield


from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.exceptions import AppError

async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "details": exc.details},
    )

app = FastAPI(title=settings.project_name, lifespan=lifespan)
app.add_exception_handler(AppError, app_error_handler)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ShedForge backend is running"}


@app.get("/health")
def root_health() -> dict[str, str]:
    return {"status": "ok"}

app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_size_bytes)
app.add_middleware(SecurityHeadersMiddleware, settings=settings)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(auth.router, prefix=f"{settings.api_prefix}/auth", tags=["auth"])
app.include_router(conflicts.router, prefix=f"{settings.api_prefix}/conflicts", tags=["conflicts"])
app.include_router(timetable.router, prefix=f"{settings.api_prefix}/timetable", tags=["timetable"])
app.include_router(programs.router, prefix=f"{settings.api_prefix}/programs", tags=["programs"])
app.include_router(courses.router, prefix=f"{settings.api_prefix}/courses", tags=["courses"])
app.include_router(feedback.router, prefix=settings.api_prefix, tags=["feedback"])
app.include_router(rooms.router, prefix=f"{settings.api_prefix}/rooms", tags=["rooms"])
app.include_router(faculty.router, prefix=f"{settings.api_prefix}/faculty", tags=["faculty"])
app.include_router(program_structure.router, prefix=settings.api_prefix, tags=["program-structure"])
app.include_router(settings_routes.router, prefix=settings.api_prefix, tags=["settings"])
app.include_router(constraints.router, prefix=settings.api_prefix, tags=["constraints"])
app.include_router(generator.router, prefix=settings.api_prefix, tags=["generator"])
app.include_router(leaves.router, prefix=settings.api_prefix, tags=["leaves"])
app.include_router(notifications.router, prefix=settings.api_prefix, tags=["notifications"])
app.include_router(issues.router, prefix=settings.api_prefix, tags=["issues"])
app.include_router(activity.router, prefix=settings.api_prefix, tags=["activity"])
app.include_router(system.router, prefix=settings.api_prefix, tags=["system"])
app.include_router(students.router, prefix=settings.api_prefix, tags=["students"])
