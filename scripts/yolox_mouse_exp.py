"""YOLOX-Nano experiment config for single-mouse detection.

This is a YOLOX experiment file (inherits from yolox.exp.Exp).
It configures YOLOX-Nano for 1-class (mouse) detection using the
COCO-formatted Kumar Lab dataset.

Usage:
    python -m yolox.tools.train \
        -f scripts/yolox_mouse_exp.py \
        -d 1 -b 16 --fp16 \
        -c yolox_nano.pth

See README or scripts/train_yolox_mouse.ps1 for full instructions.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Force standard COCO eval (no C++/ninja) before any yolox code runs.
# YOLOX's COCOeval_opt loads a JIT-compiled C++ extension that requires
# ninja + MSVC on Windows.  We patch yolox.layers so the evaluator gets
# the pure-Python pycocotools.COCOeval when it does "from yolox.layers
# import COCOeval_opt".
# ---------------------------------------------------------------------------
try:
    from pycocotools.cocoeval import COCOeval as _StandardCOCOeval  # type: ignore[import-untyped]
    import yolox.layers  # type: ignore[import-untyped]
    yolox.layers.COCOeval_opt = _StandardCOCOeval
except Exception:
    pass

from yolox.exp import Exp as MyExp  # type: ignore[import-untyped]


class Exp(MyExp):
    def __init__(self) -> None:
        super().__init__()

        # ---------- Model ----------
        self.depth = 0.33       # YOLOX-Nano depth
        self.width = 0.25       # YOLOX-Nano width
        self.depthwise = True   # YOLOX-Nano uses depthwise separable convs
        self.act = "relu"       # YOLOX-Nano uses ReLU, not SiLU
        self.num_classes = 1    # Just "mouse"

        # ---------- Input ----------
        self.input_size = (640, 640)        # Train resolution
        self.test_size = (640, 640)         # Eval/export resolution
        self.random_size = (10, 20)         # Multiscale range: 320..640 (x32)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 1.0
        self.enable_mixup = False           # Disable mixup for small dataset
        self.degrees = 10.0                 # Light rotation augmentation
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.shear = 2.0
        self.hsv_prob = 1.0                 # Colour jitter

        # ---------- Training ----------
        self.max_epoch = 80                 # Kumar dataset is small; 80 epochs
        self.warmup_epochs = 5
        self.no_aug_epochs = 10             # Last 10 epochs: no mosaic/mixup
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 5e-4
        self.momentum = 0.9

        # ---------- Data ----------
        self.data_num_workers = 4

        # Dataset paths (override via env vars or edit here)
        self.data_dir = os.environ.get(
            "YOLOX_DATA_DIR",
            "datasets/kumar_mouse_coco"
        )
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        # ---------- Evaluation ----------
        self.eval_interval = 5              # Eval every 5 epochs
        self.test_conf = 0.25
        self.nmsthre = 0.45

        # ---------- Output ----------
        self.output_dir = os.environ.get(
            "YOLOX_OUTPUT_DIR",
            "YOLOX_outputs"
        )
        self.exp_name = "yolox_mouse_nano"

    def get_model(self):
        """Build YOLOX-Nano with depthwise separable convolutions.

        The base ``Exp.get_model()`` does NOT pass ``depthwise`` to the
        backbone / head constructors.  YOLOX-Nano requires this override
        so that the architecture matches the pretrained checkpoint keys.
        """
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead  # type: ignore[import-untyped]

        def _init_yolo(M):  # noqa: N802
            import torch.nn as nn
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(_init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """Override to point at our COCO-format dataset."""
        from yolox.data import COCODataset, TrainTransform  # type: ignore[import-untyped]

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=10,
                flip_prob=0.5,
                hsv_prob=self.hsv_prob,
            ),
            name="images/train",
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_loader(self, batch_size: int, is_distributed: bool, **kwargs):
        """Override to use our val dataset."""
        import torch
        from yolox.data import COCODataset, ValTransform  # type: ignore[import-untyped]

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            preproc=ValTransform(legacy=False),
            name="images/val",
        )

        sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }

        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        return val_loader

    def get_evaluator(self, batch_size: int, is_distributed: bool, testdev: bool = False, legacy: bool = False):
        from yolox.evaluators import COCOEvaluator  # type: ignore[import-untyped]

        val_loader = self.get_eval_loader(batch_size, is_distributed)

        return COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
