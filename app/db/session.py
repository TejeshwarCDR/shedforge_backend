from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings

settings = get_settings()

connect_args = {}
if settings.database_url.startswith("postgresql+psycopg"):
    # Supabase pooler connections run through PgBouncer transaction pooling, which
    # is incompatible with psycopg prepared statements unless they are disabled.
    connect_args["prepare_threshold"] = None

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
