from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from scindra_engine.schemas import ArenaKind, AssayType


class ZoneTemplateId(str, Enum):
    OPEN_FIELD_CENTER_PERIPHERY = "OPEN_FIELD_CENTER_PERIPHERY"
    LIGHT_DARK_COMPARTMENTS = "LIGHT_DARK_COMPARTMENTS"
    EPM_ARMS = "EPM_ARMS"
    EZM_SEGMENTS = "EZM_SEGMENTS"
    Y_MAZE_ARMS = "Y_MAZE_ARMS"
    T_MAZE_ARMS = "T_MAZE_ARMS"
    BARNES_HOLES_QUADRANTS = "BARNES_HOLES_QUADRANTS"
    MWM_QUADRANTS_PLATFORM = "MWM_QUADRANTS_PLATFORM"
    NOR_OBJECT_ROIS = "NOR_OBJECT_ROIS"
    CPP_COMPARTMENTS = "CPP_COMPARTMENTS"
    THREE_CHAMBER_COMPARTMENTS = "THREE_CHAMBER_COMPARTMENTS"
    HOME_CAGE_NONE = "HOME_CAGE_NONE"


@dataclass(frozen=True)
class AssayDefinition:
    assay_type: AssayType
    default_arena_kind: ArenaKind
    allowed_arena_kinds: tuple[ArenaKind, ...]
    zone_template_id: ZoneTemplateId
    default_template_params: Mapping[str, object] = field(default_factory=dict)

