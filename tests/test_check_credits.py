
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.program import Program
from app.models.program_structure import ProgramTerm, ProgramCourse
from app.models.course import Course

def check_credits():
    db = SessionLocal()
    try:
        p = db.execute(select(Program).where(Program.code == "BTECH-CSE-2023")).scalars().first()
        if not p:
             print("Program not found")
             return
        
        terms = db.execute(select(ProgramTerm).where(ProgramTerm.program_id == p.id)).scalars().all()
        for t in terms:
            print(f"\nTERM: {t.term_number} | CREDITS REQUIRED: {t.credits_required}")
            program_courses = db.execute(select(ProgramCourse).where(ProgramCourse.program_id == p.id, ProgramCourse.term_number == t.term_number)).scalars().all()
            total_hours = 0
            for pc in program_courses:
                course = db.get(Course, pc.course_id)
                if course:
                    split_sum = course.theory_hours + course.lab_hours + course.tutorial_hours
                    print(f"  Course: {course.code} | HPW: {course.hours_per_week} | Credits: {course.credits} | Split: T{course.theory_hours}+L{course.lab_hours}+Tut{course.tutorial_hours}={split_sum}")
                    total_hours += course.hours_per_week
            print(f"TOTAL COURSE HOURS: {total_hours}")
            if total_hours != t.credits_required:
                print(f"!!! MISMATCH: {total_hours} vs {t.credits_required}")

    finally:
        db.close()

if __name__ == "__main__":
    check_credits()
