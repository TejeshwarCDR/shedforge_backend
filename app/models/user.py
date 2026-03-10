import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SAEnum, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class UserRole(str, Enum):
    admin = "admin"
    scheduler = "scheduler"
    faculty = "faculty"
    student = "student"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole, name="user_role"), nullable=False)
    program_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    department: Mapped[str | None] = mapped_column(String(200), nullable=True)
    section_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    semester_number: Mapped[int | None] = mapped_column(nullable=True)
    batch_year: Mapped[int | None] = mapped_column(nullable=True)
    roll_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
