from __future__ import annotations

import logging

import uuid

from sqlalchemy import inspect, text

from app.db.base import Base
from app.db.session import engine

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "programs": {
        "id",
        "name",
        "code",
        "department",
        "degree",
        "duration_years",
        "sections",
        "total_students",
        "default_section_capacity",
        "home_building",
        "course_mapping_enabled",
        "faculty_mapping_enabled",
        "student_mapping_enabled",
        "room_mapping_enabled",
    },
    "users": {
        "id",
        "email",
        "role",
        "section_name",
        "program_id",
        "semester_number",
        "batch_year",
        "roll_number",
    },
    "faculty": {"id", "email", "program_id", "preferred_subject_codes", "semester_preferences"},
    "courses": {
        "id",
        "code",
        "program_id",
        "semester_number",
        "batch_year",
        "theory_hours",
        "lab_hours",
        "tutorial_hours",
        "batch_segregation",
        "practical_contiguous_slots",
        "assign_faculty",
        "assign_classroom",
        "default_room_id",
        "elective_category",
    },
    "institution_settings": {"id", "academic_year", "semester_cycle"},
    "rooms": {"id", "name", "program_id"},
}


def _ensure_users_section_name_column() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "users" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("users")}
        if "section_name" in column_names:
            return
        connection.execute(text("ALTER TABLE users ADD COLUMN section_name VARCHAR(50)"))


def _resolve_default_program_id(connection: object) -> str:
    program_id = connection.execute(text("SELECT id FROM programs LIMIT 1")).scalar()
    if program_id:
        return str(program_id)

    generated_id = str(uuid.uuid4())
    connection.execute(
        text(
            """
            INSERT INTO programs (
                id,
                name,
                code,
                department,
                degree,
                duration_years,
                sections,
                total_students,
                default_section_capacity,
                home_building,
                course_mapping_enabled,
                faculty_mapping_enabled,
                student_mapping_enabled,
                room_mapping_enabled
            )
            VALUES (
                :id,
                :name,
                :code,
                :department,
                :degree,
                :duration_years,
                :sections,
                :total_students,
                :default_section_capacity,
                :home_building,
                :course_mapping_enabled,
                :faculty_mapping_enabled,
                :student_mapping_enabled,
                :room_mapping_enabled
            )
            """
        ),
        {
            "id": generated_id,
            "name": "Default Program",
            "code": "DEFAULT-PROGRAM",
            "department": "General",
            "degree": "BS",
            "duration_years": 4,
            "sections": 1,
            "total_students": 0,
            "default_section_capacity": 60,
            "home_building": None,
            "course_mapping_enabled": True,
            "faculty_mapping_enabled": True,
            "student_mapping_enabled": True,
            "room_mapping_enabled": True,
        },
    )
    return generated_id


def _ensure_users_academic_columns() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "users" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("users")}
        if "program_id" not in column_names:
            connection.execute(text("ALTER TABLE users ADD COLUMN program_id VARCHAR(36)"))
        if "semester_number" not in column_names:
            connection.execute(text("ALTER TABLE users ADD COLUMN semester_number INTEGER"))
        if "batch_year" not in column_names:
            connection.execute(text("ALTER TABLE users ADD COLUMN batch_year INTEGER"))
        if "roll_number" not in column_names:
            connection.execute(text("ALTER TABLE users ADD COLUMN roll_number VARCHAR(64)"))


def _ensure_faculty_program_id_column() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "faculty" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("faculty")}
        if "program_id" not in column_names:
            connection.execute(text("ALTER TABLE faculty ADD COLUMN program_id VARCHAR(36)"))

        default_program_id = _resolve_default_program_id(connection)
        connection.execute(
            text("UPDATE faculty SET program_id = :program_id WHERE program_id IS NULL"),
            {"program_id": default_program_id},
        )


def _ensure_course_program_id_column() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "courses" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("courses")}
        if "program_id" not in column_names:
            connection.execute(text("ALTER TABLE courses ADD COLUMN program_id VARCHAR(36)"))

        if "program_courses" in set(inspector.get_table_names()):
            rows = connection.execute(
                text("SELECT course_id, program_id FROM program_courses")
            ).fetchall()
            course_program_map: dict[str, str] = {}
            for course_id, program_id in rows:
                if course_id and program_id and course_id not in course_program_map:
                    course_program_map[str(course_id)] = str(program_id)
            for course_id, program_id in course_program_map.items():
                connection.execute(
                    text(
                        "UPDATE courses "
                        "SET program_id = :program_id "
                        "WHERE id = :course_id AND program_id IS NULL"
                    ),
                    {"program_id": program_id, "course_id": course_id},
                )

        default_program_id = _resolve_default_program_id(connection)
        connection.execute(
            text("UPDATE courses SET program_id = :program_id WHERE program_id IS NULL"),
            {"program_id": default_program_id},
        )


def _ensure_room_program_id_column() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "rooms" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("rooms")}
        if "program_id" not in column_names:
            connection.execute(text("ALTER TABLE rooms ADD COLUMN program_id VARCHAR(36)"))

        default_program_id = _resolve_default_program_id(connection)
        connection.execute(
            text("UPDATE rooms SET program_id = :program_id WHERE program_id IS NULL"),
            {"program_id": default_program_id},
        )


def _ensure_faculty_preferred_subject_codes_column() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "faculty" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("faculty")}
        if "preferred_subject_codes" in column_names:
            return

        if connection.dialect.name == "postgresql":
            connection.execute(
                text(
                    "ALTER TABLE faculty "
                    "ADD COLUMN preferred_subject_codes JSONB NOT NULL DEFAULT '[]'::jsonb"
                )
            )
            return

        connection.execute(
            text(
                "ALTER TABLE faculty "
                "ADD COLUMN preferred_subject_codes JSON NOT NULL DEFAULT '[]'"
            )
        )


def _ensure_faculty_semester_preferences_column() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "faculty" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("faculty")}
        if "semester_preferences" in column_names:
            return
        if connection.dialect.name == "postgresql":
            connection.execute(
                text(
                    "ALTER TABLE faculty "
                    "ADD COLUMN semester_preferences JSONB NOT NULL DEFAULT '{}'::jsonb"
                )
            )
            return
        connection.execute(
            text(
                "ALTER TABLE faculty "
                "ADD COLUMN semester_preferences JSON NOT NULL DEFAULT '{}'"
            )
        )


def _ensure_course_credit_split_columns() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "courses" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("courses")}
        if "semester_number" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN semester_number INTEGER NOT NULL DEFAULT 1")
            )
        if "batch_year" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN batch_year INTEGER NOT NULL DEFAULT 1")
            )
        if "theory_hours" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN theory_hours INTEGER NOT NULL DEFAULT 0")
            )
        if "lab_hours" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN lab_hours INTEGER NOT NULL DEFAULT 0")
            )
        if "tutorial_hours" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN tutorial_hours INTEGER NOT NULL DEFAULT 0")
            )
        if "batch_segregation" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN batch_segregation BOOLEAN NOT NULL DEFAULT TRUE")
            )
        if "practical_contiguous_slots" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN practical_contiguous_slots INTEGER NOT NULL DEFAULT 2")
            )
        if "assign_faculty" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN assign_faculty BOOLEAN NOT NULL DEFAULT TRUE")
            )
        if "assign_classroom" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN assign_classroom BOOLEAN NOT NULL DEFAULT TRUE")
            )
        if "default_room_id" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN default_room_id VARCHAR(36)")
            )
        if "elective_category" not in column_names:
            connection.execute(
                text("ALTER TABLE courses ADD COLUMN elective_category VARCHAR(120)")
            )
        course_type_expression = "type::text" if connection.dialect.name == "postgresql" else "CAST(type AS TEXT)"
        connection.execute(
            text(
                "UPDATE courses "
                "SET assign_faculty = FALSE, assign_classroom = FALSE "
                f"WHERE {course_type_expression} = 'elective'"
            )
        )
        connection.execute(
            text(
                "UPDATE courses "
                "SET theory_hours = CASE "
                f"WHEN {course_type_expression} = 'lab' THEN 0 "
                "WHEN theory_hours = 0 THEN hours_per_week "
                "ELSE theory_hours "
                "END, "
                "lab_hours = CASE "
                f"WHEN {course_type_expression} = 'lab' AND lab_hours = 0 THEN hours_per_week "
                "ELSE lab_hours "
                "END, "
                "tutorial_hours = COALESCE(tutorial_hours, 0) "
                "WHERE theory_hours + lab_hours + tutorial_hours = 0"
            )
        )
        connection.execute(
            text(
                "UPDATE courses "
                "SET practical_contiguous_slots = 1 "
                "WHERE COALESCE(lab_hours, 0) <= 0"
            )
        )
        connection.execute(
            text(
                "UPDATE courses "
                "SET practical_contiguous_slots = CASE "
                "WHEN practical_contiguous_slots < 1 THEN 1 "
                "WHEN practical_contiguous_slots > COALESCE(lab_hours, 0) AND COALESCE(lab_hours, 0) > 0 "
                "THEN COALESCE(lab_hours, 0) "
                "ELSE practical_contiguous_slots "
                "END"
            )
        )
        connection.execute(
            text(
                "UPDATE courses "
                "SET faculty_id = NULL "
                "WHERE assign_faculty = FALSE"
            )
        )
        connection.execute(
            text(
                "UPDATE courses "
                "SET default_room_id = NULL "
                "WHERE assign_classroom = FALSE"
            )
        )


def _ensure_institution_cycle_columns() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "institution_settings" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("institution_settings")}
        if "academic_year" not in column_names:
            connection.execute(
                text(
                    "ALTER TABLE institution_settings "
                    "ADD COLUMN academic_year VARCHAR(20) NOT NULL DEFAULT '2026-2027'"
                )
            )
        if "semester_cycle" not in column_names:
            connection.execute(
                text(
                    "ALTER TABLE institution_settings "
                    "ADD COLUMN semester_cycle VARCHAR(10) NOT NULL DEFAULT 'odd'"
                )
            )


def _ensure_program_mapping_columns() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        if "programs" not in set(inspector.get_table_names()):
            return
        column_names = {item["name"] for item in inspector.get_columns("programs")}
        if "default_section_capacity" not in column_names:
            connection.execute(
                text("ALTER TABLE programs ADD COLUMN default_section_capacity INTEGER NOT NULL DEFAULT 60")
            )
        if "home_building" not in column_names:
            connection.execute(text("ALTER TABLE programs ADD COLUMN home_building VARCHAR(200)"))
        if "course_mapping_enabled" not in column_names:
            connection.execute(
                text("ALTER TABLE programs ADD COLUMN course_mapping_enabled BOOLEAN NOT NULL DEFAULT TRUE")
            )
        if "faculty_mapping_enabled" not in column_names:
            connection.execute(
                text("ALTER TABLE programs ADD COLUMN faculty_mapping_enabled BOOLEAN NOT NULL DEFAULT TRUE")
            )
        if "student_mapping_enabled" not in column_names:
            connection.execute(
                text("ALTER TABLE programs ADD COLUMN student_mapping_enabled BOOLEAN NOT NULL DEFAULT TRUE")
            )
        if "room_mapping_enabled" not in column_names:
            connection.execute(
                text("ALTER TABLE programs ADD COLUMN room_mapping_enabled BOOLEAN NOT NULL DEFAULT TRUE")
            )


def _assert_required_columns() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        table_names = set(inspector.get_table_names())
        missing_tables = [name for name in REQUIRED_COLUMNS if name not in table_names]
        if missing_tables:
            raise RuntimeError(f"Missing required tables: {', '.join(sorted(missing_tables))}")

        missing_columns: list[str] = []
        for table_name, required in REQUIRED_COLUMNS.items():
            existing = {item["name"] for item in inspector.get_columns(table_name)}
            for column_name in sorted(required - existing):
                missing_columns.append(f"{table_name}.{column_name}")
        if missing_columns:
            raise RuntimeError(f"Missing required columns: {', '.join(missing_columns)}")


def _ensure_demo_users_exist() -> None:
    from app.core.security import get_password_hash
    with engine.begin() as connection:
        # Common Demo Password for all generated/synced accounts.
        hp = get_password_hash("ShedForge@123")
        
        # 1. Ensure a program exists for reference (optional fallback)
        program_id = connection.execute(text("SELECT id FROM programs LIMIT 1")).scalar()
        if not program_id:
            program_id = str(uuid.uuid4())
            connection.execute(
                text("INSERT INTO programs (id, name, code, department, degree, duration_years) VALUES (:id, 'Computer Science', 'CS', 'CSE', 'BS', 4)"),
                {"id": program_id}
            )
        
        # 2. Ensure Administrator exists
        connection.execute(
            text("""
                INSERT INTO users (id, name, email, hashed_password, role, is_active)
                VALUES (:id, 'ShedForge Admin', 'admin@gmail.com', :hp, 'admin', TRUE)
                ON CONFLICT (email) DO UPDATE SET hashed_password = EXCLUDED.hashed_password, is_active = TRUE
            """),
            {"id": str(uuid.uuid4()), "hp": hp}
        )

        # 3. Synchronize ALL Faculty records
        # Update all emails in the 'faculty' table to name@mail.com and ensure they have login access.
        faculty_rows = connection.execute(text("SELECT id, name, email, program_id, department FROM faculty")).fetchall()
        for f in faculty_rows:
            # Generate synthetic email: name@mail.com (lowercase, no spaces)
            new_email = f.name.lower().replace(" ", "") + "@mail.com"
            
            # Update the faculty record itself
            connection.execute(
                text("UPDATE faculty SET email = :ne WHERE id = :id"),
                {"ne": new_email, "id": f.id}
            )
            
            # Check for existing user record (by email or ID) and update/create as needed.
            # Using ID allows us to re-map existing accounts that might have changed names/emails.
            u_exists = connection.execute(text("SELECT id FROM users WHERE email = :e OR id = :id"), {"e": new_email, "id": f.id}).scalar()
            if u_exists:
                connection.execute(
                    text("UPDATE users SET name = :n, email = :e, hashed_password = :hp, role = 'faculty' WHERE id = :id"),
                    {"n": f.name, "e": new_email, "hp": hp, "id": u_exists}
                )
            else:
                connection.execute(
                    text("INSERT INTO users (id, name, email, hashed_password, role, program_id, department, is_active) VALUES (:id, :n, :e, :hp, 'faculty', :pid, :dept, TRUE)"),
                    {"id": f.id, "n": f.name, "e": new_email, "hp": hp, "pid": f.program_id, "dept": f.department}
                )

        # 4. Synchronize ALL Student records
        # In this system, students exist only in the 'users' table with role='student'.
        students = connection.execute(text("SELECT id, name FROM users WHERE role = 'student'")).fetchall()
        for s in students:
            new_email = s.name.lower().replace(" ", "") + "@mail.com"
            connection.execute(
                text("UPDATE users SET email = :e, hashed_password = :hp WHERE id = :id"),
                {"e": new_email, "hp": hp, "id": s.id}
            )

def ensure_runtime_schema_compatibility() -> None:
    try:
        # Ensure missing tables are present before additive compatibility patches.
        Base.metadata.create_all(bind=engine)
        _ensure_users_section_name_column()
        _ensure_users_academic_columns()
        _ensure_faculty_preferred_subject_codes_column()
        _ensure_faculty_semester_preferences_column()
        _ensure_faculty_program_id_column()
        _ensure_course_program_id_column()
        _ensure_room_program_id_column()
        _ensure_course_credit_split_columns()
        _ensure_institution_cycle_columns()
        _ensure_program_mapping_columns()
        _ensure_demo_users_exist()
        _assert_required_columns()
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        logger.exception("Runtime schema compatibility bootstrap failed")
        raise RuntimeError("Runtime schema compatibility bootstrap failed") from exc
