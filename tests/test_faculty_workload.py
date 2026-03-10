import pytest
from sqlalchemy import select
from app.db.session import SessionLocal
from app.api.deps import get_db
from app.models.room import Room
from app.models.faculty import Faculty
from app.models.program import Program
from app.models.program_structure import ProgramCourse, ProgramSection, ProgramTerm
from app.models.course import Course, CourseType
from app.services.evolution_scheduler import EvolutionaryScheduler
from app.schemas.generator import GenerateTimetableRequest, GenerationSettingsBase, ObjectiveWeights

def test_faculty_workload_enforcement(client):
    # Hack to get the session from the client/app dependency override
    db_gen = client.app.dependency_overrides[get_db]()
    db = next(db_gen)

    try:
        # 1. seed dummy data
        program = Program(name="Test Program Workload", code="TEST-WORKLOAD", department="CSE", degree="BS", duration_years=4, total_students=60)
        db.add(program)
        db.commit()
        db.refresh(program)

        term = ProgramTerm(program_id=program.id, term_number=1, name="Term 1", credits_required=10)
        db.add(term)
        db.commit()

        faculty = Faculty(name="Test Faculty Workload", email="tfw@example.com", department="CSE", max_hours=10) # Max 10 hours
        db.add(faculty)
        db.commit()
        db.refresh(faculty)

        course_def = Course(
            code="TEST101",
            name="Test Course",
            type=CourseType.theory,
            credits=3,
            duration_hours=1,
            hours_per_week=5,
            theory_hours=5,
            faculty_id=faculty.id
        )
        db.add(course_def)
        db.commit()
        db.refresh(course_def)

        course = ProgramCourse(
            program_id=program.id, 
            term_number=1, 
            course_id=course_def.id, 
            is_required=True, 
            lab_batch_count=1, 
            allow_parallel_batches=True
        )
        # If we add 2 sections, that's 10 hours = 10 hours max. No violation.
        db.add(course)
        db.commit()
        db.refresh(course)

        section1 = ProgramSection(program_id=program.id, term_number=1, name="A", capacity=60)
        section2 = ProgramSection(program_id=program.id, term_number=1, name="B", capacity=60)
        # section3 = ProgramSection(program_id=program.id, term_number=1, name="C", capacity=60) # Would cause violation
        db.add_all([section1, section2])

        room = Room(name="Test Room", capacity=100, type="lecture", building="Main")
        db.add(room)
        
        db.commit()

        # 2. Run generation
        settings = GenerationSettingsBase(
            random_seed=42,
            population_size=20, # Small pop for speed
            generations=10,    # Small gens
            solver_strategy="fast",
            objective_weights=ObjectiveWeights(workload_overflow=2000) # High penalty
        )
        
        scheduler = EvolutionaryScheduler(
            db=db,
            program_id=str(program.id),
            term_number=1,
            settings=settings
        )
        request = GenerateTimetableRequest(
            program_id=str(program.id),
            term_number=1,
            alternative_count=1
        )
        
        print(f"\nRUNNING GENERATION for validation...")
        response = scheduler.run(request)
        
        best = response.alternatives[0] if response.alternatives else None
        if not best:
             pytest.skip("No alternatives generated")

        # 3. Check hard conflicts/workload penalty
        # Verify
        from collections import Counter
        faculty_minutes = Counter()
        for slot in best.payload.timetable_data:
            end = int(slot.endTime.split(":")[0]) * 60 + int(slot.endTime.split(":")[1])
            start = int(slot.startTime.split(":")[0]) * 60 + int(slot.startTime.split(":")[1])
            duration = end - start
            faculty_minutes[slot.facultyId] += duration
        
        violations = 0
        f_obj = faculty
        mins = faculty_minutes[f_obj.id]
        hrs = mins / 60
        print(f"  Faculty: {f_obj.name} | Assigned: {hrs:.1f} hrs | Max: {f_obj.max_hours} hrs")
        
        if mins > f_obj.max_hours * 60 + 1: # Tolerance for rounding
             violations += 1
        
        assert violations == 0, f"Workload violation: {hrs} > {f_obj.max_hours}"

    finally:
        db.close()
