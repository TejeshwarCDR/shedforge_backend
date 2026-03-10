from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.activity import ActivityLogOut


class LabeledCount(BaseModel):
    label: str
    value: int


class DailyCountPoint(BaseModel):
    date: str
    value: int


class ResourceInventoryOut(BaseModel):
    programs: int
    program_terms: int = Field(alias="programTerms")
    program_sections: int = Field(alias="programSections")
    courses: int
    faculty: int
    rooms_total: int = Field(alias="roomsTotal")
    lecture_rooms: int = Field(alias="lectureRooms")
    lab_rooms: int = Field(alias="labRooms")
    seminar_rooms: int = Field(alias="seminarRooms")
    users_total: int = Field(alias="usersTotal")
    users_by_role: dict[str, int] = Field(default_factory=dict, alias="usersByRole")

    model_config = ConfigDict(populate_by_name=True)


class TimetableSnapshotOut(BaseModel):
    is_published: bool = Field(alias="isPublished")
    updated_at: str | None = Field(default=None, alias="updatedAt")
    total_slots: int = Field(alias="totalSlots")
    sections: int
    faculty: int
    rooms: int
    courses: int
    slots_by_day: dict[str, int] = Field(default_factory=dict, alias="slotsByDay")

    model_config = ConfigDict(populate_by_name=True)


class UtilizationSnapshotOut(BaseModel):
    room_utilization_percent: float = Field(alias="roomUtilizationPercent")
    faculty_utilization_percent: float = Field(alias="facultyUtilizationPercent")
    section_coverage_percent: float = Field(alias="sectionCoveragePercent")

    model_config = ConfigDict(populate_by_name=True)


class CapacitySnapshotOut(BaseModel):
    total_room_capacity: int = Field(alias="totalRoomCapacity")
    lecture_room_capacity: int = Field(alias="lectureRoomCapacity")
    lab_room_capacity: int = Field(alias="labRoomCapacity")
    seminar_room_capacity: int = Field(alias="seminarRoomCapacity")
    configured_section_capacity: int = Field(alias="configuredSectionCapacity")
    scheduled_student_seats: int = Field(alias="scheduledStudentSeats")

    model_config = ConfigDict(populate_by_name=True)


class ActivityAnalyticsOut(BaseModel):
    window_days: int = Field(alias="windowDays")
    total_logs: int = Field(alias="totalLogs")
    actions_last_window: int = Field(alias="actionsLastWindow")
    active_users: int = Field(alias="activeUsers")
    actions_by_day: list[DailyCountPoint] = Field(default_factory=list, alias="actionsByDay")
    top_actions: list[LabeledCount] = Field(default_factory=list, alias="topActions")
    top_entities: list[LabeledCount] = Field(default_factory=list, alias="topEntities")
    recent_logs: list[ActivityLogOut] = Field(default_factory=list, alias="recentLogs")

    model_config = ConfigDict(populate_by_name=True)


class OperationsSnapshotOut(BaseModel):
    unread_notifications: int = Field(alias="unreadNotifications")
    notifications_by_type: list[LabeledCount] = Field(default_factory=list, alias="notificationsByType")
    leaves_by_status: list[LabeledCount] = Field(default_factory=list, alias="leavesByStatus")
    issues_by_status: list[LabeledCount] = Field(default_factory=list, alias="issuesByStatus")
    feedback_by_status: list[LabeledCount] = Field(default_factory=list, alias="feedbackByStatus")

    model_config = ConfigDict(populate_by_name=True)


class AnalyticsScopeOut(BaseModel):
    mode: str
    program_id: str | None = Field(default=None, alias="programId")
    program_code: str | None = Field(default=None, alias="programCode")
    program_name: str | None = Field(default=None, alias="programName")
    timetable_scoped: bool = Field(alias="timetableScoped")
    timetable_scope_note: str = Field(alias="timetableScopeNote")

    model_config = ConfigDict(populate_by_name=True)


class MetricDefinitionOut(BaseModel):
    key: str
    label: str
    definition: str
    formula: str
    numerator: float
    denominator: float
    unit: str
    computed_value: float = Field(alias="computedValue")

    model_config = ConfigDict(populate_by_name=True)


class SystemAnalyticsOut(BaseModel):
    generated_at: str = Field(alias="generatedAt")
    scope: AnalyticsScopeOut
    inventory: ResourceInventoryOut
    timetable: TimetableSnapshotOut
    utilization: UtilizationSnapshotOut
    capacity: CapacitySnapshotOut
    activity: ActivityAnalyticsOut
    operations: OperationsSnapshotOut
    metric_definitions: list[MetricDefinitionOut] = Field(default_factory=list, alias="metricDefinitions")

    model_config = ConfigDict(populate_by_name=True)
