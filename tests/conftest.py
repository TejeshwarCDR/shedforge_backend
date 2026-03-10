import pytest
from fastapi.testclient import TestClient #gives you a fake http client that can call cyou FastAPI routes without running a real server.
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.db.base import Base
from app.main import app
from app.services.rate_limit import clear_rate_limiter


@pytest.fixture() #test client
def client(): #fake http client
    clear_rate_limiter() #resetting rate limiter state before the test starts otherwise previous tests can cause failure.
    engine = create_engine( #create isolate DB
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine) #this base contains all the SQLAlchemy models and creates the tables inside the in-memory db

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    clear_rate_limiter()
