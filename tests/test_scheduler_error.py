import pytest
from fastapi import status
from app.core.exceptions import SchedulerError, AppError
from app.api.routes.timetable import router

def test_scheduler_error_structure():
    err = SchedulerError(message="Test error", details={"foo": "bar"})
    assert err.status_code == 400
    assert err.message == "Test error"
    assert err.details == {"foo": "bar"}
    assert isinstance(err, AppError)

def test_app_error_defaults():
    err = AppError("Generic error")
    assert err.status_code == 500
    assert err.details == {}

# Note: Testing the global exception handler requires `TestClient` and the full app.
# We can do that in an integration test, but unit testing the error class logic is sufficient here.
