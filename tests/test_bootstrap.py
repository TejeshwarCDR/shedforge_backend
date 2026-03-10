import pytest

from app.db import bootstrap


def _raise_error(message: str):
    raise RuntimeError(message)


def test_runtime_schema_bootstrap_raises_on_validation_failure(monkeypatch):
    monkeypatch.setattr(bootstrap.Base.metadata, "create_all", lambda bind: None)
    monkeypatch.setattr(bootstrap, "_ensure_users_section_name_column", lambda: None)
    monkeypatch.setattr(bootstrap, "_ensure_faculty_preferred_subject_codes_column", lambda: None)
    monkeypatch.setattr(
        bootstrap,
        "_assert_required_columns",
        lambda: _raise_error("missing required schema"),
    )

    with pytest.raises(RuntimeError, match="Runtime schema compatibility bootstrap failed"):
        bootstrap.ensure_runtime_schema_compatibility()
