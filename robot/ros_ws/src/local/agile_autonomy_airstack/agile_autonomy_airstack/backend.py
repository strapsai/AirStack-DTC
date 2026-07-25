from dataclasses import dataclass


@dataclass(frozen=True)
class BackendPoint:
    time_from_start: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BackendCandidate:
    candidate_id: str
    cost: float
    frame_mode: str
    points: list


@dataclass(frozen=True)
class BackendResult:
    healthy: bool
    reason: str
    candidates: list


class MockAgileBackend:
    def __init__(self, horizon_sec, dt_sec, forward_speed_mps, speed_limit_mps, frame_mode):
        self.horizon_sec = float(horizon_sec)
        self.dt_sec = float(dt_sec)
        self.forward_speed_mps = float(forward_speed_mps)
        self.speed_limit_mps = float(speed_limit_mps)
        self.frame_mode = frame_mode
        self.sequence = 0

    def generate(self):
        self.sequence += 1
        if self.forward_speed_mps > self.speed_limit_mps:
            speed = 0.0
            reason = "mock_speed_exceeds_limit_stationary_candidate"
        else:
            speed = max(0.0, self.forward_speed_mps)
            reason = "mock"

        num_points = max(2, int(round(self.horizon_sec / max(self.dt_sec, 1e-6))))
        points = []
        for index in range(num_points):
            if num_points == 1:
                t = self.horizon_sec
            else:
                t = self.horizon_sec * float(index) / float(num_points - 1)
            points.append(BackendPoint(t, speed * t, 0.0, 0.0))

        candidate = BackendCandidate(
            candidate_id=f"mock_forward_{self.sequence:06d}",
            cost=0.0,
            frame_mode=self.frame_mode,
            points=points,
        )
        return BackendResult(True, reason, [candidate])


class ExternalAgileBackend:
    def __init__(self, backend_config=""):
        self.backend_config = backend_config

    def generate(self):
        if not self.backend_config:
            return BackendResult(False, "external_backend_not_configured", [])
        return BackendResult(False, "external_backend_extension_point_unimplemented", [])


def create_backend(
    backend_type,
    horizon_sec,
    dt_sec,
    forward_speed_mps,
    speed_limit_mps,
    frame_mode,
    backend_config="",
):
    if backend_type == "mock":
        return MockAgileBackend(horizon_sec, dt_sec, forward_speed_mps, speed_limit_mps, frame_mode)
    if backend_type == "external":
        return ExternalAgileBackend(backend_config)
    return ExternalAgileBackend("")
