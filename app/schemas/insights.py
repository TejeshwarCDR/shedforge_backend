from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.timetable import OfficialTimetablePayload


class TimetableConflict(BaseModel):
    id: str
    type: Literal[
        "faculty-overlap",
        "room-overlap",
        "section-overlap",
        "elective-overlap",
        "capacity",
        "availability",
        "course-faculty-inconsistency",
    ]
    severity: Literal["high", "medium", "low"]
    description: str
    affected_slots: list[str] = Field(default_factory=list, alias="affectedSlots")
    resolution: str
    resolved: bool = False
    decision: Literal["yes", "no"] | None = None
    resolution_mode: Literal["auto", "manual", "ignored", "pending"] | None = Field(
        default=None,
        alias="resolutionMode",
    )
    decision_note: str | None = Field(default=None, alias="decisionNote")

    model_config = ConfigDict(populate_by_name=True)


class ConflictDecisionIn(BaseModel):
    decision: Literal["yes", "no"]
    note: str | None = Field(default=None, max_length=500)


class ConflictDecisionOut(BaseModel):
    conflict_id: str
    decision: Literal["yes", "no"]
    resolved: bool
    message: str
    published_version_label: str | None = None


class ConstraintStatus(BaseModel):
    name: str
    description: str
    satisfaction: float
    status: Literal["satisfied", "partial", "violated"]


class WorkloadChartEntry(BaseModel):
    id: str
    name: str
    full_name: str = Field(alias="fullName")
    department: str
    workload: float
    max: float
    overloaded: bool

    model_config = ConfigDict(populate_by_name=True)


class DailyWorkloadEntry(BaseModel):
    day: str
    loads: dict[str, float]
    total: float


class PerformanceTrendEntry(BaseModel):
    semester: str
    satisfaction: float
    conflicts: int


class OptimizationSummary(BaseModel):
    constraint_satisfaction: float = Field(alias="constraintSatisfaction")
    conflicts_detected: int = Field(alias="conflictsDetected")
    optimization_technique: str = Field(alias="optimizationTechnique")
    alternatives_generated: int = Field(alias="alternativesGenerated")
    last_generated: str | None = Field(default=None, alias="lastGenerated")
    total_iterations: int = Field(alias="totalIterations")
    compute_time: str = Field(alias="computeTime")

    model_config = ConfigDict(populate_by_name=True)


class TimetableAnalytics(BaseModel):
    optimization_summary: OptimizationSummary = Field(alias="optimizationSummary")
    constraint_data: list[ConstraintStatus] = Field(default_factory=list, alias="constraintData")
    workload_chart_data: list[WorkloadChartEntry] = Field(default_factory=list, alias="workloadChartData")
    daily_workload_data: list[DailyWorkloadEntry] = Field(default_factory=list, alias="dailyWorkloadData")
    performance_trend_data: list[PerformanceTrendEntry] = Field(default_factory=list, alias="performanceTrendData")

    model_config = ConfigDict(populate_by_name=True)


class TimetableConflictReviewIn(BaseModel):
    payload: OfficialTimetablePayload | None = None


class TimetableConflictReviewOut(BaseModel):
    source: Literal["official", "provided"]
    auto_resolved_conflicts: list[TimetableConflict] = Field(default_factory=list, alias="autoResolvedConflicts")
    manually_resolved_conflicts: list[TimetableConflict] = Field(default_factory=list, alias="manuallyResolvedConflicts")
    ignored_conflicts: list[TimetableConflict] = Field(default_factory=list, alias="ignoredConflicts")
    pending_conflicts: list[TimetableConflict] = Field(default_factory=list, alias="pendingConflicts")
    unresolved_required_count: int = Field(alias="unresolvedRequiredCount")
    unresolved_hard_count: int = Field(alias="unresolvedHardCount")
    constraint_mismatches: list[str] = Field(default_factory=list, alias="constraintMismatches")
    can_publish: bool = Field(alias="canPublish")
    can_publish_anyway: bool = Field(default=True, alias="canPublishAnyway")

    model_config = ConfigDict(populate_by_name=True)


class TimetableConflictResolveAllIn(BaseModel):
    payload: OfficialTimetablePayload | None = None
    scope: Literal["hard", "all"] = "hard"
    promote_official: bool | None = Field(default=None, alias="promoteOfficial")
    note: str | None = Field(default=None, max_length=500)

    model_config = ConfigDict(populate_by_name=True)


class TimetableConflictResolveAllOut(BaseModel):
    source: Literal["official", "provided"]
    resolved_payload: OfficialTimetablePayload = Field(alias="resolvedPayload")
    resolved_count: int = Field(alias="resolvedCount")
    remaining_conflicts: list[TimetableConflict] = Field(default_factory=list, alias="remainingConflicts")
    auto_resolved_conflicts: list[TimetableConflict] = Field(default_factory=list, alias="autoResolvedConflicts")
    constraint_mismatches: list[str] = Field(default_factory=list, alias="constraintMismatches")
    promoted_version_label: str | None = Field(default=None, alias="promotedVersionLabel")

    model_config = ConfigDict(populate_by_name=True)
