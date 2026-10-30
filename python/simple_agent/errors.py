class TurnInterrupted(Exception):
    """A completed node yielded to new input; this is not a business failure."""


class RunError(Exception):
    def __init__(self, code, message, *, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def record(self):
        return {"code": self.code, "message": str(self), "details": self.details}
