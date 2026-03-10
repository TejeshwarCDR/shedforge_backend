class AppError(Exception):
    """Base class for all application exceptions."""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class SchedulerError(AppError):
    """Raised when the scheduler encounters a logical error or invalid state."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400, details=details)

class ResourceNotFoundError(AppError):
    """Raised when a requested resource is not found."""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(f"{resource_type} with id {resource_id} not found", status_code=404)

class ConfigurationError(AppError):
    """Raised when system configuration is invalid."""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)
