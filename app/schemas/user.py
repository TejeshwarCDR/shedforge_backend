from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from app.models.user import UserRole


class UserBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    role: UserRole
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    department: str | None = None
    section_name: str | None = Field(default=None, min_length=1, max_length=50)
    semester_number: int | None = Field(default=None, ge=1, le=20)
    batch_year: int | None = Field(default=None, ge=1, le=8)
    roll_number: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Name cannot be empty")
        return trimmed

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("department")
    @classmethod
    def normalize_department(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("section_name")
    @classmethod
    def normalize_section_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("roll_number")
    @classmethod
    def normalize_roll_number(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip().upper()
        return trimmed or None


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)
    preferred_subject_codes: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("preferred_subject_codes")
    @classmethod
    def normalize_preferred_subject_codes(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized_codes: list[str] = []
        for item in value:
            code = item.strip().upper()
            if not code:
                continue
            if len(code) > 50:
                raise ValueError("Preferred subject code length cannot exceed 50 characters")
            if code in seen:
                continue
            seen.add(code)
            normalized_codes.append(code)
        return normalized_codes

    @model_validator(mode="after")
    def validate_role_specific_requirements(self) -> "UserCreate":
        if self.role == UserRole.student:
            if not self.section_name:
                raise ValueError("section_name is required for student registration")
            if self.semester_number is None:
                raise ValueError("semester_number is required for student registration")
        else:
            self.section_name = None
            self.semester_number = None
            self.batch_year = None
            self.roll_number = None
        return self


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: UserRole | None = None

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class LoginOtpRequest(UserLogin):
    pass


class LoginOtpChallengeOut(BaseModel):
    challenge_id: str
    email: EmailStr
    expires_in_seconds: int = Field(ge=1)
    message: str
    otp_hint: str | None = None


class LoginOtpVerify(BaseModel):
    challenge_id: str = Field(min_length=1, max_length=36)
    otp_code: str = Field(pattern=r"^\d{6}$")


class UserOut(UserBase):
    id: str

    model_config = {"from_attributes": True}


class StudentListOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    program_id: str | None = None
    department: str | None = None
    section_name: str | None = None
    semester_number: int | None = None
    batch_year: int | None = None
    roll_number: str | None = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class StudentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    department: str | None = Field(default=None, min_length=1, max_length=200)
    section_name: str = Field(min_length=1, max_length=50)
    semester_number: int = Field(ge=1, le=20)
    batch_year: int | None = Field(default=None, ge=1, le=8)
    roll_number: str | None = Field(default=None, min_length=1, max_length=64)
    is_active: bool = True

    @field_validator("name")
    @classmethod
    def normalize_student_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Name cannot be empty")
        return trimmed

    @field_validator("email")
    @classmethod
    def normalize_student_email(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("department")
    @classmethod
    def normalize_student_department(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("section_name")
    @classmethod
    def normalize_student_section_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("section_name cannot be empty")
        return trimmed

    @field_validator("roll_number")
    @classmethod
    def normalize_student_roll_number(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip().upper()
        return trimmed or None


class StudentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    email: EmailStr | None = None
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    department: str | None = Field(default=None, min_length=1, max_length=200)
    section_name: str | None = Field(default=None, min_length=1, max_length=50)
    semester_number: int | None = Field(default=None, ge=1, le=20)
    batch_year: int | None = Field(default=None, ge=1, le=8)
    roll_number: str | None = Field(default=None, min_length=1, max_length=64)
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def normalize_optional_student_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("email")
    @classmethod
    def normalize_optional_student_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().lower()

    @field_validator("department")
    @classmethod
    def normalize_optional_student_department(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("section_name")
    @classmethod
    def normalize_optional_student_section_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("roll_number")
    @classmethod
    def normalize_optional_student_roll_number(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip().upper()
        return trimmed or None


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserOut
