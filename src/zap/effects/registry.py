"""Effect registry: effect classes register under a name that config bindings use."""

from __future__ import annotations

import difflib
from typing import Any

from zap.effects.base import Effect

EFFECTS: dict[str, type[Effect]] = {}


def register(name: str):
    def deco(cls: type[Effect]) -> type[Effect]:
        cls.name = name
        EFFECTS[name] = cls
        return cls

    return deco


def load_builtin() -> None:
    """Import the built-in effect modules so their @register decorators run."""
    from zap.effects import charge, cinema, cutaway, money, outro, sad, thanos  # noqa: F401


def create(effect: str, /, **params: Any) -> Effect:
    """Instantiate a registered effect. Positional-only so params may include ``name=``."""
    if effect not in EFFECTS:
        hint = difflib.get_close_matches(effect, list(EFFECTS), n=1)
        raise KeyError(f"unknown effect '{effect}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
    return EFFECTS[effect](**params)
