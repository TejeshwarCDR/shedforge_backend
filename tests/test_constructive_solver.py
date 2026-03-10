from unittest.mock import MagicMock
import pytest
from collections import defaultdict
from app.services.evolution_scheduler import EvolutionaryScheduler, BlockRequest, PlacementOption
from app.models.course import Course, CourseType
from app.models.room import Room, RoomType
from app.models.faculty import Faculty
from app.schemas.generator import GenerationSettingsBase

class MockScheduler(EvolutionaryScheduler):
    def __init__(self):
        # Bypass __init__ to avoid DB calls
        self.settings = GenerationSettingsBase()
        self.random = MagicMock()
        self.block_requests = []
        self.rooms = {}
        self.faculty = {}
        self.day_slots = {}
        # Initialize required attributes for constructive solver
        self.fixed_genes = {}
        self.eval_cache = {}
        self.schedule_policy = MagicMock()
        self.schedule_policy.period_minutes = 60
        self.expected_section_minutes = 0
    
    def _parallel_lab_group_key(self, request):
        return None

    def _parallel_lab_signature(self, option):
        return (str(option.day), int(option.start_index))

    def _filter_option_indices_by_signatures(self, req, candidate_indices, signatures):
        return candidate_indices

    def _spread_option_indices_by_day(self, req, candidate_indices):
        return candidate_indices

    def _single_faculty_required(self, course_id):
        return False


def create_mock_request(req_id, course_id, student_count, block_size=1, is_lab=False, options_count=10, options=None):
    if options is None:
        options = []
        for i in range(options_count):
            options.append(PlacementOption(
                day="Monday",
                start_index=i,
                room_id="r1",
                faculty_id="f1"
            ))
    
    return BlockRequest(
        request_id=req_id,
        course_id=course_id,
        course_code=f"C{req_id}",
        section="A",
        batch=None,
        student_count=student_count,
        primary_faculty_id="f1",
        preferred_faculty_ids=(),
        block_size=block_size,
        is_lab=is_lab,
        session_type="lab" if is_lab else "theory",
        allow_parallel_batches=False,
        room_candidate_ids=("r1",),
        options=tuple(options)
    )

def test_request_priority_order_lab_first():
    scheduler = MockScheduler()
    req1 = create_mock_request(0, "c1", 30, block_size=1, is_lab=False)
    req2 = create_mock_request(1, "c2", 30, block_size=2, is_lab=True)
    scheduler.block_requests = [req1, req2]
    order = scheduler._request_priority_order()
    assert order == [1, 0]

def test_request_priority_order_large_blocks_first():
    scheduler = MockScheduler()
    req1 = create_mock_request(0, "c1", 30, block_size=1, is_lab=False)
    req2 = create_mock_request(1, "c2", 30, block_size=3, is_lab=False)
    scheduler.block_requests = [req1, req2]
    assert scheduler._request_priority_order() == [1, 0]

def test_request_priority_order_few_options_first():
    scheduler = MockScheduler()
    req1 = create_mock_request(0, "c1", 30, options_count=10)
    req2 = create_mock_request(1, "c2", 30, options_count=2)
    scheduler.block_requests = [req1, req2]
    assert scheduler._request_priority_order() == [1, 0]

def test_constructive_individual_picks_best_fit():
    scheduler = MockScheduler()
    
    # Mock rooms
    scheduler.rooms = {
        "room_opt0": Room(id="room_opt0", name="R0", capacity=35, type=RoomType.lecture),
        "room_opt1": Room(id="room_opt1", name="R1", capacity=100, type=RoomType.lecture),
        "room_opt2": Room(id="room_opt2", name="R2", capacity=32, type=RoomType.lecture),
    }

    # Prepare Options
    opts = [
        PlacementOption(day="Mon", start_index=0, room_id="room_opt0", faculty_id="f1"), # Waste 5
        PlacementOption(day="Mon", start_index=1, room_id="room_opt1", faculty_id="f1"), # Waste 70
        PlacementOption(day="Mon", start_index=2, room_id="room_opt2", faculty_id="f1"), # Waste 2 (Best)
    ]

    req = create_mock_request(0, "c1", 30, options=opts)
    scheduler.block_requests = [req]
    
    # Mock Helpers
    scheduler._option_candidate_indices = MagicMock(return_value=[0, 1, 2])
    scheduler._is_immediately_conflict_free = MagicMock(return_value=True)
    scheduler._incremental_option_penalty = MagicMock(return_value=(0, 0.0))
    
    genes = scheduler._constructive_individual(randomized=False)
    
    assert genes[0] == 2  # Best fit option

def test_constructive_individual_avoids_hard_conflicts_fallback():
    scheduler = MockScheduler()
    
    scheduler.rooms = {
        "r1": Room(id="r1", name="R1", capacity=40, type=RoomType.lecture)
    }
    opts = [
        PlacementOption(day="Mon", start_index=0, room_id="r1", faculty_id="f1"), # Conflicted
        PlacementOption(day="Mon", start_index=1, room_id="r1", faculty_id="f1"), # Free
    ]
    req = create_mock_request(0, "c1", 30, options=opts)
    scheduler.block_requests = [req]
    
    scheduler._option_candidate_indices = MagicMock(return_value=[0, 1])
    
    # Mock conflict check: 0 is conflicted, 1 is free
    def fake_conflict_check(**kwargs):
        return kwargs["option_index"] == 1
        
    scheduler._is_immediately_conflict_free = MagicMock(side_effect=fake_conflict_check)
    scheduler._incremental_option_penalty = MagicMock(return_value=(0, 0.0))

    genes = scheduler._constructive_individual(randomized=False)
    
    assert genes[0] == 1
