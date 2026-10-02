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


class ParameterUndefinedError(ValidationError):
    """A bound parameter name does not resolve in the circuit's expressions."""

    code = "PARAM_UNDEFINED"


class ParameterArrayLengthMismatchError(ValidationError):
    """Parameter value arrays do not all have the same length."""

    code = "PARAM_ARRAY_LENGTH_MISMATCH"


class ParameterArrayEmptyError(ValidationError):
    """A parameter value array is empty."""

    code = "PARAM_ARRAY_EMPTY"


class ParameterValueInvalidError(ValidationError):
    """A parameter value is not a finite number."""

    code = "PARAM_VALUE_INVALID"


class ShotsInvalidError(ValidationError):
    """`shots` is not a positive integer in a parameterized batch request."""

    code = "SHOTS_INVALID"


class BatchLimitExceededError(ValidationError):
    """`scenarios * shots` exceeds the service limit; no job is created."""

    code = "BATCH_LIMIT_EXCEEDED"
