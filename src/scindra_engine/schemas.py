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


class ArenaROIConfig(BaseModel):
    """Restrict tracking to a region of interest (arena mask).

    When enabled, the segmentation mask is AND-ed with the arena mask so that
    only detections inside the arena are considered.  This prevents the tracker
    from latching onto high-contrast objects outside the arena (e.g. door
    handles, labels on the cage).

    Specify the ROI either via ``mask_path`` (a binary image where white pixels
    mark the valid region) or via ``kind`` + ``params`` (a geometric shape).
    All coordinate/size parameters are given in **original video resolution**;
    they are automatically scaled when ``downsample_factor`` is used.
    """

    enabled: bool = False
    mask_path: str | None = Field(
        default=None,
        description="Path to a binary mask image (white pixels = inside arena).",
    )
    kind: Literal["CIRCLE", "RECT"] | None = Field(
        default=None,
        description="Geometric shape for the ROI.",
    )
    params: dict[str, float] | None = Field(
        default=None,
        description=(
            "Shape parameters in original video coordinates. "
            "CIRCLE: {center_x, center_y, radius}. "
            "RECT: {x, y, w, h}."
        ),
    )

    @model_validator(mode="after")
    def validate_roi_source(self) -> ArenaROIConfig:
        if not self.enabled:
            return self
        if self.mask_path is None and self.kind is None:
            raise ValueError(
                "Arena ROI is enabled but neither mask_path nor kind is set."
            )
        if self.kind is not None and self.params is None:
            raise ValueError(
                "Arena ROI kind is set but params is missing."
            )
        if self.kind == "CIRCLE":
            required = {"center_x", "center_y", "radius"}
            if self.params is None or not required.issubset(self.params):
                raise ValueError(
                    f"CIRCLE ROI requires params: {required}"
                )
        elif self.kind == "RECT":
            required = {"x", "y", "w", "h"}
            if self.params is None or not required.issubset(self.params):
                raise ValueError(
                    f"RECT ROI requires params: {required}"
                )
        return self


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
    background_model: Literal["none", "median_n", "mog2"] = "median_n"
    background_n: int = 25
    mog2_history: int = Field(default=500, ge=1, description="MOG2 history length (frames).")
    mog2_var_threshold: float = Field(default=16.0, ge=0.0, description="MOG2 variance threshold.")
    mog2_detect_shadows: bool = Field(default=True, description="MOG2 shadow detection.")


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
    open_ksize: int = 5
    close_ksize: int = 7
    erode_iters: int = 0
    dilate_iters: int = 0


class MotionMaskConfig(BaseModel):
    """Frame-difference motion mask to suppress static foreground features.

    When enabled, only pixels that changed in the last ``history_len``
    frames are kept as foreground.  This eliminates static high-contrast
    objects (maze ring, screws, labels) that survive background subtraction.
    """

    enabled: bool = True
    history_len: int = Field(
        default=10, ge=1,
        description="Number of past frames kept for motion comparison.",
    )
    threshold: int = Field(
        default=15, ge=1, le=255,
        description="Pixel-change threshold to qualify as motion.",
    )
    dilate_ksize: int = Field(
        default=7, ge=1,
        description="Dilation kernel size applied to the raw motion mask.",
    )
    dilate_iters: int = Field(
        default=3, ge=0,
        description="Number of dilation iterations.",
    )


class TrackingConfig(BaseModel):
    min_area_px: int = 50
    max_area_px: int = 100000
    max_jump_px: float = 80.0
    smoothing: Literal["none", "ema"] = "none"
    ema_alpha: float = 0.3
    # --- adaptive area ---
    adaptive_area: bool = Field(
        default=True,
        description="Narrow the area filter around a running median of recent detections.",
    )
    adaptive_area_ratio: float = Field(
        default=3.0, ge=1.0,
        description="Allowed area range: [median / ratio, median * ratio].",
    )
    adaptive_area_history: int = Field(
        default=30, ge=3,
        description="Number of recent detections used for the running median.",
    )
    # --- Kalman filter ---
    use_kalman: bool = Field(
        default=True,
        description="Use a Kalman filter for prediction-based gating and scoring.",
    )
    kalman_process_noise: float = Field(
        default=4.0, ge=0.0,
        description="Kalman process-noise covariance diagonal (pixel^2).",
    )
    kalman_measurement_noise: float = Field(
        default=4.0, ge=0.0,
        description="Kalman measurement-noise covariance diagonal (pixel^2).",
    )
    kalman_gate_sigma: float = Field(
        default=4.0, ge=0.0,
        description="Mahalanobis-distance gate in sigma units.",
    )
    kalman_coast_frames: int = Field(
        default=5, ge=0,
        description=(
            "Maximum consecutive frames to coast on Kalman prediction when "
            "no detection is found.  0 = disable coasting."
        ),
    )


class KeyFrameInterpolationConfig(BaseModel):
    """Post-processing pass that identifies high-confidence key frames and
    interpolates between them to fill NO_DETECTION gaps and correct drift.

    A frame qualifies as a key frame when its confidence meets
    ``min_confidence`` and it has no ``AMBIGUOUS_TARGET`` flag.  Between two
    key frames, NO_DETECTION frames (and optionally points that deviate too
    far from the interpolated path) are replaced with interpolated positions.
    """

    enabled: bool = False
    min_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for a frame to qualify as a key frame.",
    )
    max_gap_frames: int = Field(
        default=30,
        ge=1,
        description="Maximum number of consecutive non-key frames to interpolate across.",
    )
    max_deviation_px: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Points deviating more than this from the interpolated path are "
            "replaced.  None = only fill NO_DETECTION gaps."
        ),
    )
    method: Literal["linear", "cubic"] = Field(
        default="linear",
        description="Interpolation method between key frames.",
    )


class ChromaFilterConfig(BaseModel):
    """Chrominance-based shadow suppression.

    Shadows change luminance but not chrominance.  By requiring *either* a
    minimum chrominance difference *or* a large luminance difference from the
    background, shadow blobs are suppressed while the mouse is preserved.

    A pixel is kept as foreground when:
      ``chroma_diff > threshold``  OR  ``luma_diff > luma_threshold``

    This means a dark mouse on a light floor (same hue, huge brightness
    change) is kept, while a shadow on the floor (same hue, small brightness
    change) is suppressed.

    Operates in CIE-Lab colour space.
    """

    enabled: bool = True
    threshold: int = Field(
        default=12, ge=1, le=255,
        description="Minimum chrominance delta (in Lab a/b units) to qualify as foreground.",
    )
    luma_threshold: int = Field(
        default=40, ge=1, le=255,
        description=(
            "Minimum luminance delta (Lab L channel) to qualify as foreground "
            "even when chrominance is similar.  Prevents filtering out dark "
            "objects on light backgrounds (or vice versa) that share a similar "
            "hue.  Set higher than the typical shadow luminance drop (~20-30)."
        ),
    )


class DetectorConfig(BaseModel):
    """Configuration for the optional YOLOX detector-assisted ROI tracking."""

    enabled: bool = Field(
        default=False,
        description="Enable detector-assisted ROI tracking.",
    )
    backend: Literal["YOLOX_ONNX"] = "YOLOX_ONNX"
    model_path: str | None = Field(
        default=None,
        description="Explicit path to the ONNX model file.",
    )
    min_score: float = Field(
        default=0.35, ge=0.0, le=1.0,
        description="Minimum detector score to accept a detection for ROI.",
    )
    every_n_frames: int = Field(
        default=15, ge=1,
        description="Run the detector every N frames.",
    )
    reacquire_on_low_tracking_conf: bool = Field(
        default=True,
        description="Re-run detector when tracking confidence drops.",
    )
    roi_padding_px: int = Field(
        default=60, ge=0,
        description="Pixels to pad around the detector bounding box for the ROI (used when roi_padding_ratio is not set).",
    )
    roi_padding_ratio: float | None = Field(
        default=None, ge=0.0,
        description="Padding as fraction of bbox size: padding on each side = ratio * min(bbox_width, bbox_height). When set, overrides roi_padding_px. E.g. 0.5 = 50%% of smaller dimension.",
    )
    roi_padding_scale: float | None = Field(
        default=None, ge=1.0,
        description="Alternative: scale factor for ROI around bbox (e.g. 1.5x). Takes precedence over roi_padding_ratio/roi_padding_px when set.",
    )
    max_roi_jump_px: float = Field(
        default=200.0, ge=0.0,
        description="Max allowed jump in ROI center (pixels); larger triggers hysteresis. Used when max_roi_jump_ratio is not set.",
    )
    max_roi_jump_ratio: float | None = Field(
        default=None, ge=0.0,
        description="Max allowed jump as fraction of bbox size: threshold = ratio * max(bbox_width, bbox_height). When set, overrides max_roi_jump_px for scale-invariant behavior.",
    )
    precompute_roi_parallel: bool = Field(
        default=False,
        description="When True, run detector in a pre-pass to build a per-frame ROI schedule, then use parallel chunked tracking with those ROIs.",
    )
    fallback_to_classical_full_frame: bool = Field(
        default=True,
        description="Fall back to classical full-frame tracking when detector unavailable.",
    )
    write_detector_debug_frames: bool = Field(
        default=True,
        description="Write sampled debug frames with detector bbox overlay.",
    )
    detector_debug_frame_count: int = Field(
        default=10, ge=1, le=50,
        description="Number of detector debug frames to write.",
    )


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
    motion_mask: MotionMaskConfig = Field(
        default_factory=MotionMaskConfig,
        description="Motion mask to suppress static foreground features.",
    )
    chroma_filter: ChromaFilterConfig = Field(
        default_factory=ChromaFilterConfig,
        description="Chrominance-based shadow suppression.",
    )
    arena_roi: ArenaROIConfig = Field(
        default_factory=ArenaROIConfig,
        description="Optional arena region-of-interest mask to restrict tracking.",
    )
    key_frame_interpolation: KeyFrameInterpolationConfig = Field(
        default_factory=KeyFrameInterpolationConfig,
        description="Optional key-frame interpolation post-processing.",
    )
    detector: DetectorConfig = Field(
        default_factory=DetectorConfig,
        description="Optional detector-assisted ROI tracking.",
    )
    progress_interval: int = Field(default=50, ge=1)
    ambiguity_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    shadow_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    parallel_workers: int | None = Field(default=None, ge=1, description="Number of parallel workers for frame processing (None = auto)")
    chunk_size: int = Field(default=200, ge=1, description="Number of frames per chunk for parallel processing")
    downsample_factor: float | None = Field(default=None, ge=1.0, description="Downsample frames by this factor before processing (e.g., 2.0 = half resolution). Coordinates are scaled back to original resolution.")
    debug_mode: bool = Field(
        default=False,
        description="When True, write debug frames showing centroid blobs (detected/excluded) and use sequential processing.",
    )
    debug_frame_interval: int = Field(
        default=30, ge=1,
        description="When debug_mode is True, write a debug frame every N frames.",
    )
    debug_max_frames: int | None = Field(
        default=100, ge=1,
        description="When debug_mode is True, cap the number of debug frames written. None = no cap.",
    )


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
