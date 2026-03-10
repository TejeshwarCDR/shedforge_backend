
import pytest
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.program import Program
from app.models.program_structure import ProgramTerm
from app.services.evolution_scheduler import EvolutionaryScheduler
from app.schemas.generator import GenerationSettingsBase, ObjectiveWeights
from fastapi import HTTPException

def test_curriculum_mismatch_detection():
    db = SessionLocal()
    try:
        program = db.execute(select(Program).where(Program.code == "BTECH-CSE-2023")).scalars().first()
        settings = GenerationSettingsBase(
            random_seed=42,
            population_size=20,
            generations=10,
            solver_strategy="fast",
            objective_weights=ObjectiveWeights()
        )
        
        print(f"\nTESTING FLEXIBLE ALIGNMENT FOR {program.code} TERM 1...")
        scheduler = EvolutionaryScheduler(
            db=db,
            program_id=str(program.id),
            term_number=1,
            settings=settings
        )
        assert scheduler.expected_section_minutes == 29 * scheduler.schedule_policy.period_minutes
        print(f"SUCCESS: Term 1 validated with {scheduler.expected_section_minutes//60} hours as expected.")

    finally:
        db.close()

if __name__ == "__main__":
    test_curriculum_mismatch_detection()
