from __future__ import annotations

from typing import Mapping

from scindra_engine.schemas import AssayType

from .types import AssayDefinition, ZoneTemplateId


registry: Mapping[AssayType, AssayDefinition] = {
    "OPEN_FIELD": AssayDefinition(
        assay_type="OPEN_FIELD",
        default_arena_kind="RECT",
        allowed_arena_kinds=("RECT", "CIRCLE"),
        zone_template_id=ZoneTemplateId.OPEN_FIELD_CENTER_PERIPHERY,
    ),
    "LIGHT_DARK_BOX": AssayDefinition(
        assay_type="LIGHT_DARK_BOX",
        default_arena_kind="COMPARTMENTS",
        allowed_arena_kinds=("COMPARTMENTS",),
        zone_template_id=ZoneTemplateId.LIGHT_DARK_COMPARTMENTS,
    ),
    "EPM": AssayDefinition(
        assay_type="EPM",
        default_arena_kind="PLUS",
        allowed_arena_kinds=("PLUS",),
        zone_template_id=ZoneTemplateId.EPM_ARMS,
    ),
    "EZM": AssayDefinition(
        assay_type="EZM",
        default_arena_kind="ANNULUS",
        allowed_arena_kinds=("ANNULUS",),
        zone_template_id=ZoneTemplateId.EZM_SEGMENTS,
    ),
    "Y_MAZE": AssayDefinition(
        assay_type="Y_MAZE",
        default_arena_kind="Y",
        allowed_arena_kinds=("Y",),
        zone_template_id=ZoneTemplateId.Y_MAZE_ARMS,
    ),
    "T_MAZE": AssayDefinition(
        assay_type="T_MAZE",
        default_arena_kind="T",
        allowed_arena_kinds=("T",),
        zone_template_id=ZoneTemplateId.T_MAZE_ARMS,
    ),
    "BARNES_MAZE": AssayDefinition(
        assay_type="BARNES_MAZE",
        default_arena_kind="RADIAL",
        allowed_arena_kinds=("RADIAL",),
        zone_template_id=ZoneTemplateId.BARNES_HOLES_QUADRANTS,
    ),
    "MORRIS_WATER_MAZE": AssayDefinition(
        assay_type="MORRIS_WATER_MAZE",
        default_arena_kind="CIRCLE",
        allowed_arena_kinds=("CIRCLE",),
        zone_template_id=ZoneTemplateId.MWM_QUADRANTS_PLATFORM,
    ),
    "NOVEL_OBJECT_RECOGNITION": AssayDefinition(
        assay_type="NOVEL_OBJECT_RECOGNITION",
        default_arena_kind="RECT",
        allowed_arena_kinds=("RECT", "CIRCLE"),
        zone_template_id=ZoneTemplateId.NOR_OBJECT_ROIS,
    ),
    "CONDITIONED_PLACE_PREFERENCE": AssayDefinition(
        assay_type="CONDITIONED_PLACE_PREFERENCE",
        default_arena_kind="COMPARTMENTS",
        allowed_arena_kinds=("COMPARTMENTS",),
        zone_template_id=ZoneTemplateId.CPP_COMPARTMENTS,
    ),
    "THREE_CHAMBER_SOCIAL": AssayDefinition(
        assay_type="THREE_CHAMBER_SOCIAL",
        default_arena_kind="COMPARTMENTS",
        allowed_arena_kinds=("COMPARTMENTS",),
        zone_template_id=ZoneTemplateId.THREE_CHAMBER_COMPARTMENTS,
    ),
    "HOME_CAGE_LOCOMOTION": AssayDefinition(
        assay_type="HOME_CAGE_LOCOMOTION",
        default_arena_kind="RECT",
        allowed_arena_kinds=("RECT",),
        zone_template_id=ZoneTemplateId.HOME_CAGE_NONE,
    ),
}


def list_assays() -> list[AssayType]:
    return sorted(registry.keys())


def get_assay(assay_type: AssayType) -> AssayDefinition:
    return registry[assay_type]

