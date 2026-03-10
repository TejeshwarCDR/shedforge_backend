from __future__ import annotations


def designation_workload_cap(designation: str | None) -> int:
    normalized = (designation or "").strip().lower()
    if "assistant professor" in normalized:
        return 16
    if "associate professor" in normalized or "professor" in normalized:
        return 14
    return 16


def constrained_max_hours(designation: str | None, requested_max_hours: int | None) -> int:
    cap = designation_workload_cap(designation)
    if requested_max_hours is None:
        return cap
    if requested_max_hours < 1:
        return 1
    return min(requested_max_hours, cap)
