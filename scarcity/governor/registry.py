"""
Subsystem registry for DRG actuators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SubsystemHandle:
    name: str
    handle: Any

    def call(self, method: str, *args, **kwargs) -> bool:
        func = getattr(self.handle, method, None)
        if callable(func):
            func(*args, **kwargs)
            return True
        return False


class SubsystemRegistry:
    def __init__(self):
        self._handles: Dict[str, SubsystemHandle] = {}

    def register(self, name: str, handle: Any) -> None:
        self._handles[name] = SubsystemHandle(name, handle)

    def get(self, name: str) -> Optional[SubsystemHandle]:
        return self._handles.get(name)

    def all(self) -> Dict[str, SubsystemHandle]:
        return dict(self._handles)

