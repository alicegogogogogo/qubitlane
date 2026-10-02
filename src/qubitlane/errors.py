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


class BatchError(ValidationError):
    """A rejected parameter-sweep submission; still a 400 validation failure."""


class ShotsInvalidError(BatchError):
    code = "SHOTS_INVALID"


class ParamUndefinedError(BatchError):
    code = "PARAM_UNDEFINED"


class ParamArrayLengthMismatchError(BatchError):
    code = "PARAM_ARRAY_LENGTH_MISMATCH"


class ParamArrayEmptyError(BatchError):
    code = "PARAM_ARRAY_EMPTY"


class ParamValueInvalidError(BatchError):
    code = "PARAM_VALUE_INVALID"


class BatchLimitExceededError(BatchError):
    code = "BATCH_LIMIT_EXCEEDED"
