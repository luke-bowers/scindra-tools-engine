from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# Enums / Literals
AssayType = Literal[
    "OPEN_FIELD",
    "LIGHT_DARK_BOX",
    "EPM",
    "EZM",
    "Y_MAZE",
    "T_MAZE",
    "BARNES_MAZE",
    "MORRIS_WATER_MAZE",
    "NOVEL_OBJECT_RECOGNITION",
    "CONDITIONED_PLACE_PREFERENCE",
    "THREE_CHAMBER_SOCIAL",
    "HOME_CAGE_LOCOMOTION",
]

AssaySelectionMode = Literal["AUTO", "MANUAL"]

ArenaKind = Literal[
    "RECT",
    "CIRCLE",
    "ANNULUS",
    "PLUS",
    "Y",
    "T",
    "RADIAL",
    "COMPARTMENTS",
]

TrackerBackend = Literal["CLASSICAL"]


# Nested models for AnalysisConfig
class AssayConfig(BaseModel):
    selection_mode: AssaySelectionMode = "AUTO"
    assay_type: AssayType | None = None
    min_confidence_to_auto_accept: float = Field(default=0.85, ge=0.0, le=1.0)


class VideoConfig(BaseModel):
    path: str
    fps_override: float | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> VideoConfig:
        if (
            self.start_time_s is not None
            and self.end_time_s is not None
            and self.start_time_s >= self.end_time_s
        ):
            raise ValueError(
                "start_time_s must be less than end_time_s when both are provided"
            )
        return self


class ArenaDetectionConfig(BaseModel):
    enabled: bool = True
    mode: Literal["AUTO", "MANUAL"] = "AUTO"
    manual_arena_kind: ArenaKind | None = None
    manual_params: dict[str, object] | None = None
    min_confidence_to_auto_accept: float = Field(default=0.85, ge=0.0, le=1.0)


class ZonesConfig(BaseModel):
    enabled: bool = True
    mode: Literal["AUTO_TEMPLATE", "MANUAL"] = "AUTO_TEMPLATE"
    template_params: dict[str, object] = Field(default_factory=dict)


class PreprocessingConfig(BaseModel):
    grayscale: bool = True
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    gamma: float | None = None
    denoise: Literal["none", "gaussian", "bilateral"] = "none"
    illumination_correction: Literal["none", "rolling_ball", "morph_open"] = "none"
    background_model: Literal["none", "median_n"] = "median_n"
    background_n: int = 25


class SegmentationConfig(BaseModel):
    threshold: Literal["otsu", "adaptive", "manual"] = "otsu"
    manual_value: int | None = None
    adaptive_block_size: int = 35
    adaptive_C: int = 2
    invert: bool = False

    @field_validator("manual_value")
    @classmethod
    def validate_manual_value(cls, v: int | None) -> int | None:
        if v is not None and (v < 0 or v > 255):
            raise ValueError("manual_value must be between 0 and 255")
        return v

    @field_validator("adaptive_block_size")
    @classmethod
    def validate_adaptive_block_size(cls, v: int) -> int:
        if v < 3:
            raise ValueError("adaptive_block_size must be >= 3")
        if v % 2 == 0:
            raise ValueError("adaptive_block_size must be odd")
        return v

    @model_validator(mode="after")
    def validate_manual_threshold(self) -> SegmentationConfig:
        if self.threshold == "manual" and self.manual_value is None:
            raise ValueError("manual_value is required when threshold is 'manual'")
        return self


class MorphologyConfig(BaseModel):
    open_ksize: int = 3
    close_ksize: int = 3
    erode_iters: int = 0
    dilate_iters: int = 0


class TrackingConfig(BaseModel):
    min_area_px: int = 50
    max_area_px: int = 100000
    max_jump_px: float = 80.0
    smoothing: Literal["none", "ema"] = "none"
    ema_alpha: float = 0.3


class QCConfig(BaseModel):
    min_track_coverage: float = 0.8
    max_jump_rate: float = 0.1
    min_mean_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class TrackCentroidConfig(BaseModel):
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    segmentation: SegmentationConfig = Field(
        # Background subtraction yields a bright foreground on dark background,
        # so the default should keep foreground as white for connected components.
        default_factory=lambda: SegmentationConfig(invert=False)
    )
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    progress_interval: int = Field(default=50, ge=1)
    ambiguity_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    shadow_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class OutputsConfig(BaseModel):
    out_dir: str
    write_overlay_video: bool = True
    write_debug_frames: bool = True
    debug_frame_count: int = Field(default=10, ge=1, le=50)


class MetadataConfig(BaseModel):
    operator: str | None = None
    lab: str | None = None
    notes: str | None = None


# Main AnalysisConfig model
class AnalysisConfig(BaseModel):
    assay: AssayConfig
    video: VideoConfig
    arena_detection: ArenaDetectionConfig = Field(default_factory=ArenaDetectionConfig)
    zones: ZonesConfig = Field(default_factory=ZonesConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    outputs: OutputsConfig
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)

    @model_validator(mode="after")
    def validate_manual_assay_type(self) -> AnalysisConfig:
        if (
            self.assay.selection_mode == "MANUAL"
            and self.assay.assay_type is None
        ):
            raise ValueError(
                "assay_type is required when selection_mode is 'MANUAL'"
            )
        return self


# ArtifactManifest model
class InputFile(BaseModel):
    path: str
    sha256: str
    bytes: int


class ConfigFile(BaseModel):
    path: str
    sha256: str


class OutputFile(BaseModel):
    kind: str
    path: str
    sha256: str
    bytes: int


class ArtifactManifest(BaseModel):
    engine_version: str
    git_commit: str | None = None
    run_id: str
    started_at: str
    finished_at: str
    input_files: list[InputFile]
    config: ConfigFile
    outputs: list[OutputFile]
    qc_metrics: dict[str, float | int | str | bool]
    summary_metrics: dict[str, float | int | str | bool]
    warnings: list[str]
    needs_review: bool
    review_reasons: list[str]
    confidence: dict[str, float]
