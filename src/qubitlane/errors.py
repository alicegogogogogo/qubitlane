class QubitLaneError(Exception):
    code = "internal_error"
    status = 500


class ValidationError(QubitLaneError):
    code = "validation_error"
    status = 400


class ParseError(ValidationError):
    """A circuit body could not be parsed; reported with a 1-based line number."""

    code = "validation_error"
    status = 400


class NotFoundError(QubitLaneError):
    code = "not_found"
    status = 404


class ConflictError(QubitLaneError):
    code = "conflict"
    status = 409
