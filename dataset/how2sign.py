import torch
import numpy as np
import json
from typing import Dict, List, Union, Any
from pathlib import Path


class How2Sign(torch.utils.data.Dataset):
    """
    Dataset class for the How2Sign sign language dataset.

    This class mirrors the Phoenix14T dataset style as closely as possible,
    but reads samples from a JSON manifest (with splits) instead of *.npy annotations.
    """

    def __init__(
        self,
        manifest: str,
        how2sign_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = "train",  # train | val | test
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = "",
        spatiotemporal_postfix: Union[str, List[str]] = "",
    ):
        super().__init__()

        self.manifest = Path(manifest)
        self.how2sign_root = Path(how2sign_root)
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix

        # Validate inputs (identical logic to Phoenix14T)
        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")

        # Load manifest (instead of .npy)
        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")

        with open(self.manifest, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "splits" not in data:
            raise KeyError("Manifest missing 'splits' key")

        if mode not in data["splits"]:
            raise KeyError(f"Split '{mode}' not found in manifest")

        self.data = data["splits"][mode]  # keep name `self.data` like Phoenix14T

        # Set up directory paths (identical pattern to Phoenix14T)
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode

        # Validate that key directories exist (same checks)
        self._validate_directories()

        if not self.how2sign_root.exists():
            raise FileNotFoundError(f"how2sign_root not found: {self.how2sign_root}")

    def _validate_directories(self) -> None:
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")

        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}")

    def _feat_id_from_item(self, item: Dict[str, Any]) -> str:
        """
        Build feature id for How2Sign samples.

        Minimal, deterministic formatting: <sentence_name>_<start_str>
        """
        sentence_name = item.get("sentence_name")
        if not sentence_name:
            sentence_name = Path(item["video_relpath"]).stem

        start = item.get("start_realigned", item.get("start"))
        if start is None:
            return sentence_name

        start_str = f"{float(start):.2f}".rstrip("0").rstrip(".")
        return f"{sentence_name}_{start_str}"

    def _load_spatial_features(self, file_id: str) -> torch.Tensor:
        feat_path = self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Spatial feature file not found: {feat_path}")
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal_features(self, file_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(self.spatiotemporal_postfix, str):
            glor_path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not glor_path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {glor_path}")
            return torch.tensor(np.load(glor_path))
        else:
            feats = []
            for postfix in self.spatiotemporal_postfix:
                path = self.spatiotemporal_dir / f"{file_id}{postfix}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
                feats.append(torch.tensor(np.load(path)))
            return feats

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        file_id = self._feat_id_from_item(item)

        pixel_value = None
        glor_value = None

        # Load spatial features if enabled (same style as Phoenix14T)
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])

        # Load spatiotemporal features if enabled (same style as Phoenix14T)
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]

        video_path = self.how2sign_root / item["video_relpath"]

        result = {
            "pixel_value": pixel_value,
            "glor_value": glor_value,
            "bool_mask_pos": None,
            "text": self._normalize_text(item.get("sentence", "")),
            "gloss": "",  # gloss-free
            "id": file_id,
            "num_frames": len(pixel_value) if pixel_value is not None else 0,
            "vid_path": str(video_path),
            "lang": "English",
        }

        # Keep extra fields minimal; store original like Phoenix14T
        result["original_info"] = item

        return result

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip()
        if text and not text.endswith("."):
            text = f"{text}."
        return text

    def __len__(self) -> int:
        # mirror Phoenix14T behavior (they do len(self.data) - 1)
        return len(self.data) - 1

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch