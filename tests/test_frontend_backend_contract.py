from fastapi.routing import APIRoute, APIWebSocketRoute

from app.main import app


EXPECTED_HTTP_ENDPOINTS: set[tuple[str, str]] = {
    ("/api/activity/logs", "get"),
    ("/api/auth/login", "post"),
    ("/api/auth/login/request-otp", "post"),
    ("/api/auth/login/verify-otp", "post"),
    ("/api/auth/logout", "post"),
    ("/api/auth/me", "get"),
    ("/api/auth/password/change", "post"),
    ("/api/auth/password/forgot", "post"),
    ("/api/auth/password/reset", "post"),
    ("/api/auth/register", "post"),
    ("/api/constraints/semesters", "get"),
    ("/api/constraints/semesters/{term_number}", "delete"),
    ("/api/constraints/semesters/{term_number}", "get"),
    ("/api/constraints/semesters/{term_number}", "put"),
    ("/api/courses/", "get"),
    ("/api/courses/", "post"),
    ("/api/courses/{course_id}", "delete"),
    ("/api/courses/{course_id}", "put"),
    ("/api/faculty/", "get"),
    ("/api/faculty/", "post"),
    ("/api/faculty/me", "get"),
    ("/api/faculty/substitutes/suggestions", "get"),
    ("/api/faculty/{faculty_id}", "delete"),
    ("/api/faculty/{faculty_id}", "put"),
    ("/api/feedback", "get"),
    ("/api/feedback", "post"),
    ("/api/feedback/{feedback_id}", "get"),
    ("/api/feedback/{feedback_id}", "put"),
    ("/api/feedback/{feedback_id}/messages", "post"),
    ("/api/health", "get"),
    ("/api/health/live", "get"),
    ("/api/health/ready", "get"),
    ("/api/issues", "get"),
    ("/api/issues", "post"),
    ("/api/issues/{issue_id}", "put"),
    ("/api/leaves", "get"),
    ("/api/leaves", "post"),
    ("/api/leaves/{leave_id}/status", "put"),
    ("/api/leaves/{leave_id}/substitute", "post"),
    ("/api/notifications", "get"),
    ("/api/notifications/read-all", "post"),
    ("/api/notifications/{notification_id}/read", "post"),
    ("/api/programs/", "get"),
    ("/api/programs/", "post"),
    ("/api/programs/{program_id}", "delete"),
    ("/api/programs/{program_id}", "put"),
    ("/api/programs/{program_id}/courses", "get"),
    ("/api/programs/{program_id}/courses", "post"),
    ("/api/programs/{program_id}/courses/{program_course_id}", "delete"),
    ("/api/programs/{program_id}/elective-groups", "get"),
    ("/api/programs/{program_id}/elective-groups", "post"),
    ("/api/programs/{program_id}/elective-groups/{group_id}", "delete"),
    ("/api/programs/{program_id}/elective-groups/{group_id}", "put"),
    ("/api/programs/{program_id}/sections", "get"),
    ("/api/programs/{program_id}/sections", "post"),
    ("/api/programs/{program_id}/sections/{section_id}", "delete"),
    ("/api/programs/{program_id}/shared-lecture-groups", "get"),
    ("/api/programs/{program_id}/shared-lecture-groups", "post"),
    ("/api/programs/{program_id}/shared-lecture-groups/{group_id}", "delete"),
    ("/api/programs/{program_id}/shared-lecture-groups/{group_id}", "put"),
    ("/api/programs/{program_id}/terms", "get"),
    ("/api/programs/{program_id}/terms", "post"),
    ("/api/programs/{program_id}/terms/{term_id}", "delete"),
    ("/api/rooms/", "get"),
    ("/api/rooms/", "post"),
    ("/api/rooms/{room_id}", "delete"),
    ("/api/rooms/{room_id}", "put"),
    ("/api/settings/schedule-policy", "get"),
    ("/api/settings/schedule-policy", "put"),
    ("/api/settings/smtp/config", "get"),
    ("/api/settings/smtp/test", "post"),
    ("/api/settings/working-hours", "get"),
    ("/api/settings/working-hours", "put"),
    ("/api/students", "get"),
    ("/api/system/analytics", "get"),
    ("/api/system/backup", "post"),
    ("/api/system/info", "get"),
    ("/api/timetable/analytics", "get"),
    ("/api/timetable/conflicts", "get"),
    ("/api/timetable/conflicts/analyze", "post"),
    ("/api/timetable/conflicts/{conflict_id}/decision", "post"),
    ("/api/timetable/generate", "post"),
    ("/api/timetable/generate-cycle", "post"),
    ("/api/timetable/generation-settings", "get"),
    ("/api/timetable/generation-settings", "put"),
    ("/api/timetable/locks", "get"),
    ("/api/timetable/locks", "post"),
    ("/api/timetable/locks/{lock_id}", "delete"),
    ("/api/timetable/official", "get"),
    ("/api/timetable/official", "put"),
    ("/api/timetable/publish-offline", "post"),
    ("/api/timetable/publish-offline/all", "post"),
    ("/api/timetable/reevaluation/events", "get"),
    ("/api/timetable/reevaluation/run", "post"),
    ("/api/timetable/trends", "get"),
    ("/api/timetable/versions", "get"),
    ("/api/timetable/versions/compare", "get"),
}


def test_frontend_consumed_http_endpoints_are_exposed_by_backend() -> None:
    available = {
        (route.path, method.lower())
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
        if method not in {"HEAD", "OPTIONS"}
    }

    missing = sorted(EXPECTED_HTTP_ENDPOINTS - available)
    assert not missing, f"Frontend API contract mismatch. Missing backend endpoints: {missing}"


def test_frontend_consumed_notifications_websocket_exists() -> None:
    websocket_paths = {
        route.path
        for route in app.routes
        if isinstance(route, APIWebSocketRoute)
    }
    assert "/api/notifications/ws" in websocket_paths
