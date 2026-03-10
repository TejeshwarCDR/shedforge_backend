import pytest
from sqlalchemy import select
from app.models.program import Program
from app.models.room import Room, RoomType
from app.db.session import SessionLocal

def test_inspect_rooms():
    db = SessionLocal()
    try:
        rooms = db.execute(select(Room)).scalars().all()
        print(f"\nTOTAL ROOMS: {len(rooms)}")
        for r in rooms:
            print(f"ROOM: {r.name} | TYPE: {r.type} | CAP: {r.capacity}")
        
        lecture_count = sum(1 for r in rooms if r.type == RoomType.lecture)
        lab_count = sum(1 for r in rooms if r.type == RoomType.lab)
        seminar_count = sum(1 for r in rooms if r.type == RoomType.seminar)
        
        print(f"\nCOUNTS -> Lecture: {lecture_count}, Lab: {lab_count}, Seminar: {seminar_count}")
        
        # Simulate scheduler's logic
        student_count = 60 # Typical section size
        import hashlib
        
        def room_tiebreak(room: Room, seed: str) -> str:
            return hashlib.blake2b(f"{seed}|{room.id}".encode()).hexdigest()

        def select_mock(room_candidates, student_count, seed):
            ranked = sorted(
                room_candidates,
                key=lambda room: (
                    room.capacity < student_count,
                    abs(room.capacity - student_count),
                    room_tiebreak(room, seed),
                ),
            )
            return ranked[:20]

        lecture_rooms = [r for r in rooms if r.type == RoomType.lecture]
        
        print("\nSECTIONS FOR BTECH-CSE-2023 TERM 5:")
        from app.models.program_structure import ProgramSection
        p = db.execute(select(Program).where(Program.code == "BTECH-CSE-2023")).scalars().first()
        sections = db.execute(select(ProgramSection).where(ProgramSection.program_id == p.id, ProgramSection.term_number == 5)).scalars().all()
        for s in sections:
            print(f"SECTION: {s.name} | CAP: {s.capacity}")
            
    finally:
        db.close()
