from app.models.activity_log import ActivityLog  # noqa: F401
from app.models.course import Course, CourseType  # noqa: F401
from app.models.feedback import (  # noqa: F401
    FeedbackCategory,
    FeedbackItem,
    FeedbackMessage,
    FeedbackPriority,
    FeedbackStatus,
)
from app.models.faculty import Faculty  # noqa: F401
from app.models.institution_settings import InstitutionSettings  # noqa: F401
from app.models.login_otp import LoginOtpChallenge  # noqa: F401
from app.models.leave_request import LeaveRequest, LeaveStatus, LeaveType  # noqa: F401
from app.models.leave_substitute_offer import (  # noqa: F401
    LeaveSubstituteOffer,
    LeaveSubstituteOfferStatus,
)
from app.models.leave_substitute_assignment import LeaveSubstituteAssignment  # noqa: F401
from app.models.notification import Notification, NotificationType  # noqa: F401
from app.models.password_reset import PasswordResetToken  # noqa: F401
from app.models.program import Program, ProgramDegree  # noqa: F401
from app.models.program_constraint import ProgramConstraint  # noqa: F401
from app.models.program_structure import (  # noqa: F401
    ElectiveConflictPolicy,
    ProgramCourse,
    ProgramElectiveGroup,
    ProgramElectiveGroupMember,
    ProgramSection,
    ProgramSharedLectureGroup,
    ProgramSharedLectureGroupMember,
    ProgramTerm,
)
from app.models.room import Room, RoomType  # noqa: F401
from app.models.semester_constraint import SemesterConstraint  # noqa: F401
from app.models.timetable_generation import (  # noqa: F401
    ReevaluationStatus,
    TimetableGenerationSettings,
    TimetableReevaluationEvent,
    TimetableSlotLock,
)
from app.models.timetable_conflict_decision import (  # noqa: F401
    ConflictDecision,
    TimetableConflictDecision,
)
from app.models.timetable_issue import (  # noqa: F401
    IssueCategory,
    IssueMessage,
    IssueStatus,
    TimetableIssue,
)
from app.models.timetable import OfficialTimetable  # noqa: F401
from app.models.timetable_change_request import (  # noqa: F401
    TimetableChangeRequest,
    TimetableChangeRequestStatus,
)
from app.models.timetable_version import TimetableVersion  # noqa: F401
from app.models.user import User, UserRole  # noqa: F401
