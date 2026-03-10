import time
import sys
import os
# Add backend to path so we can import app modules if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock
from app.services.evolution_scheduler import EvolutionaryScheduler, BlockRequest, PlacementOption
from app.models.room import Room, RoomType
from app.models.faculty import Faculty
from app.schemas.generator import GenerationSettingsBase

class MockSlot:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class BenchmarkScheduler(EvolutionaryScheduler):
    def __init__(self, num_courses=50, sections_per_course=2):
        # Bypass DB init
        self.settings = GenerationSettingsBase()
        self.random = MagicMock()
        self.db = MagicMock()
        self.job_id = 999
        
        # Initialize containers
        self.rooms = {}
        self.faculty = {}
        self.block_requests = []
        self.fixed_genes = {}
        self.eval_cache = {}
        self.schedule_policy = MagicMock()
        self.schedule_policy.period_minutes = 60
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.time_slots = ["09:00", "10:00", "11:00", "12:00", "14:00", "15:00", "16:00"]
        self.expected_section_minutes = 0
        self.semester_constraint = None
        
        # Initialize day_slots
        self.day_slots = {}
        for day in self.days:
            slots = []
            for t_idx, time_str in enumerate(self.time_slots):
                h, m = map(int, time_str.split(":"))
                start_min = h * 60 + m
                end_min = start_min + 60
                slots.append(MockSlot(start_min, end_min))
            self.day_slots[day] = slots

        # Missing attributes from EvolutionaryScheduler
        self.reserved_resource_slots_by_day = {}
        self.request_indices_by_course = {}
        self.request_indices_by_course_section = {}
        self.single_faculty_required_by_course = {}
        self.common_faculty_candidates_by_course_section = {}
        self.common_faculty_candidates_by_course = {}
        self.faculty_windows = {}
        self.room_windows = {}

        self.courses = {}
        self.sections = []
        self.program_courses = []
        self.elective_overlap_pairs = set()
        self.shared_lecture_sections_by_course = {}
        self.faculty_preferred_subject_codes = {}
        
        self._setup_data(num_courses, sections_per_course)
        
        # Calculate priorities strictly after setup data
        self.option_priority_indices = self._build_option_priority_indices()

    def _setup_data(self, num_courses, sections_per_course):
        # 1. Rooms (10 rooms)
        for i in range(10):
            r_id = f"R{i}"
            self.rooms[r_id] = Room(id=r_id, name=f"Room {i}", capacity=60, type=RoomType.lecture)

        # 2. Faculty (20 faculty)
        for i in range(20):
            f_id = f"F{i}"
            self.faculty[f_id] = Faculty(id=f_id, name=f"Faculty {i}", department="Dept", email=f"f{i}@example.com", max_hours=20)

        # 3. Requests
        req_id_counter = 0
        for c in range(num_courses):
            c_id = f"C{c}"
            self.courses[c_id] = MagicMock(id=c_id, type="theory") # Mock course object
            for s in range(sections_per_course):
                # Each section needs 3 theory slots
                for slot_idx in range(3):
                    req_id = req_id_counter
                    req_id_counter += 1
                    
                    # Create options (All slots in all rooms)
                    options = []
                    for d_idx, day in enumerate(self.days):
                        for t_idx, time in enumerate(self.time_slots):
                            # Start index roughly maps to flattened time structure
                            start_index = d_idx * len(self.time_slots) + t_idx
                            for r_id in self.rooms:
                                options.append(PlacementOption(
                                    day=day,
                                    start_index=start_index,
                                    room_id=r_id,
                                    faculty_id=f"F{c % 20}" # Cycle faculty
                                ))
                    
                    self.block_requests.append(BlockRequest(
                        request_id=req_id,
                        course_id=f"C{c}",
                        course_code=f"CODE{c}",
                        section=f"S{s}",
                        batch=None,
                        student_count=40,
                        primary_faculty_id=f"F{c % 20}",
                        preferred_faculty_ids=(),
                        block_size=1,
                        is_lab=False,
                        session_type="theory",
                        allow_parallel_batches=False,
                        room_candidate_ids=list(self.rooms.keys()),
                        options=tuple(options)
                    ))
    
    # Mock helpers
    def _parallel_lab_group_key(self, request): return None
    def _parallel_lab_signature(self, option): return (str(option.day), int(option.start_index))
    def _filter_option_indices_by_signatures(self, req, candidate_indices, signatures): return candidate_indices
    def _spread_option_indices_by_day(self, req, candidate_indices): return candidate_indices
    def _single_faculty_required(self, course_id): return False
    
    # We need to implement basic conflict checks since `_is_immediately_conflict_free` relies on them
    # But for benchmark, we can use the actual logic OR a simplified one.
    # To test REAL performance, we should ideally use the real logic, but without DB.
    # The real `_is_immediately_conflict_free` checks `self.genes` (the individual being built).
    
    # We will let the REAL `_is_immediately_conflict_free` run, but we need to ensure 
    # `self.schedule_policy` and other things it uses are set.
    # `_is_immediately_conflict_free` uses `self.rooms`, `self.faculty`.
    # It assumes `option` has `start_index`.
    
    # Since we are inheriting from EvolutionaryScheduler, we can use its methods if we didn't override them.
    # `_constructive_individual` calls `_is_immediately_conflict_free`.

def test_performance_constructive_solver():
    # Setup: 50 courses * 2 sections * 3 slots = 300 slots to schedule
    # This is a medium-sized department schedule.
    scheduler = BenchmarkScheduler(num_courses=40, sections_per_course=2)
    
    print(f"\nStarting benchmark with {len(scheduler.block_requests)} block requests...")
    
    start_time = time.time()
    
    # Run the constructive heuristic
    # This is the core logic we want to benchmark.
    individual = scheduler._constructive_individual(randomized=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Scheduling completed in {duration:.4f} seconds.")
    
    # Assert performance target (e.g., < 5 seconds for this workload)
    # The user requirement was < 10 seconds for standard dataset.
    assert duration < 5.0, f"Scheduler took too long: {duration:.4f}s"
    
    # Basic validity check
    assert len(individual) == len(scheduler.block_requests)
    assert all(g != -1 for g in individual)

if __name__ == "__main__":
    test_performance_constructive_solver()
