import pytest
import logging
from sqlalchemy import select
from app.models.program import Program
from app.models.room import Room
from app.services.evolution_scheduler import EvolutionaryScheduler
from app.schemas.generator import GenerateTimetableRequest, GenerationSettingsBase
from app.db.session import SessionLocal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_gen")

def test_run_diagonal_gen():
    db = SessionLocal()
    try:
        # Find BTECH-CSE-2023 program
        program = db.execute(select(Program).where(Program.code == "BTECH-CSE-2023")).scalars().first()
        if not program:
            print("BTECH-CSE-2023 not found, falling back to any program")
            program = db.execute(select(Program)).scalars().first()
        
        if not program:
            print("No programs found!")
            return
        
        # Find term with most courses
        from app.models.program_structure import ProgramCourse
        from sqlalchemy import func
        term_counts = db.execute(
            select(ProgramCourse.term_number, func.count(ProgramCourse.id))
            .where(ProgramCourse.program_id == program.id)
            .group_by(ProgramCourse.term_number)
        ).all()
        
        if not term_counts:
            print("No courses found for program!")
            return
            
        term_number, count = max(term_counts, key=lambda x: x[1])
        program_id = program.id
        
        print(f"\nTESTING GENERATION FOR Program: {program.code} | Term: {term_number} | Course Count: {count}")
        
        settings = GenerationSettingsBase()
        # Make GA very fast for testing
        settings.population_size = 20
        settings.generations = 5
        
        request = GenerateTimetableRequest(
            program_id=program_id,
            term_number=term_number,
            alternative_count=1
        )
        
        scheduler = EvolutionaryScheduler(
            db=db,
            program_id=program_id,
            term_number=term_number,
            settings=settings
        )
        
        # Run generation
        result = scheduler.run(request)
        
        print(f"\nGENERATION FINISHED | Alternatives: {len(result.alternatives)}")
        if result.alternatives:
            best = result.alternatives[0]
            print(f"Best Alt -> Fitness: {best.fitness}, Hard Conflicts: {best.hard_conflicts}")
            
            # Count A102 occurrences in the payload
            a102_count = 0
            room_names = []
            
            room_map = {r.id: r.name for r in best.payload.room_data}
            
            for entry in best.payload.timetable_data:
                room_name = room_map.get(entry.roomId, "Unknown")
                if room_name == "A102":
                    a102_count += 1
                room_names.append(room_name)
            
            print(f"A102 Usage Count: {a102_count} / {len(best.payload.timetable_data)}")
            
            # Distinct rooms used
            distinct_rooms = sorted(list(set(room_names)))
            print(f"Distinct Rooms Used Count: {len(distinct_rooms)}")
            
            # Per-room usage counts
            from collections import Counter
            counts = Counter(room_names)
            print(f"Top 5 Rooms by Usage: {counts.most_common(5)}")
            
            # Type distribution
            type_counts = Counter([e.sessionType for e in best.payload.timetable_data])
            print(f"Session Type Distribution: {type_counts}")
            
            # Sample sessions
            for entry in best.payload.timetable_data[:15]:
                print(f"  Sample: {entry.courseId} | Sec: {entry.section} | Day: {entry.day} | StartTime: {entry.startTime} | Room: {room_map.get(entry.roomId)}")
            
            # Check for direct overlaps in the payload
            room_occ_check = {}
            section_occ_check = {}
            room_overlaps = 0
            section_overlaps = 0
            for entry in best.payload.timetable_data:
                # Room overlap
                rk = (entry.day, entry.startTime, entry.roomId)
                if rk in room_occ_check:
                    room_overlaps += 1
                room_occ_check[rk] = entry.courseId
                
                # Section overlap
                sk = (entry.day, entry.startTime, entry.section)
                if sk in section_occ_check:
                    section_overlaps += 1
                section_occ_check[sk] = entry.courseId
            
            print(f"\nDETECTED OVERLAPS IN PAYLOAD:")
            print(f"Room Overlaps: {room_overlaps}")
            print(f"Section Overlaps: {section_overlaps}")
            
            if a102_count > 5 and len(distinct_rooms) < 10:
                 print("\nCRITICAL: ROOM CLUSTERING DETECTED!")
        else:
            print("NO ALTERNATIVES GENERATED!")
            
    finally:
        db.close()
