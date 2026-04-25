"""Health/readiness state machine for the query node."""

from __future__ import annotations

from enum import Enum


class NodePhase(str, Enum):
    STARTING = "starting"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    UPDATING = "updating"
    FATAL = "fatal"


class ReadinessState:
    """Tracks the query node's readiness phase."""

    def __init__(self) -> None:
        self._phase = NodePhase.STARTING

    @property
    def phase(self) -> NodePhase:
        return self._phase

    @property
    def is_healthy(self) -> bool:
        return self._phase != NodePhase.FATAL

    @property
    def is_ready(self) -> bool:
        return self._phase in (NodePhase.READY, NodePhase.UPDATING)

    def set_phase(self, phase: NodePhase) -> None:
        self._phase = phase
