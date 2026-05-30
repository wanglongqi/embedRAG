"""Health and readiness state machine for the query node.

Tracks the node lifecycle phase (STARTING → DOWNLOADING → LOADING → READY
→ FATAL) so that load balancers and orchestrators can distinguish between
transient unavailability and permanent failure.

The ``ReadinessState`` class drives the /health and /readiness endpoints.
"""

from __future__ import annotations

from enum import StrEnum


class NodePhase(StrEnum):
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
        """Advance the node to a new lifecycle phase.

        Args:
            phase: The target ``NodePhase`` to transition to.
        """
        self._phase = phase
