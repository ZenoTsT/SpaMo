import torch
import numpy as np
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Union


class How2Sign(torch.utils.data.Dataset):
    """
    How2Sign dataset for SpaMo.

    Feature naming convention (as produced by extraction scripts):

      ViT:
        <sentence_name>_<start>.npy
        e.g. --7E2sU6zP4_10-5-rgb_front_18.25.npy

      MAE:
        <sentence_name>_<start>_overlap-8.npy
    """

    def __init__(
        self,
        manifest: str,
        how2sign_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = "train",  # train | val | test
        spatial: bool = True,
        spatiotemporal: bool = True,
        spatial_postfix: str = "",
        spatiotemporal_postfix: Union[str, List[str]] = "_overlap-8",
        lang: str = "English",
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
        self.lang = lang

        if not (self.spatial or self.spatiotemporal):
            raise ValueError("At least one of spatial or spatiotemporal must be True")

        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")

        with open(self.manifest, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "splits" not in data:
            raise KeyError("Manifest missing 'splits' key")

        if mode not in data["splits"]:
            raise KeyError(f"Split '{mode}' not found in manifest")

        self.items = data["splits"][mode]

        # Expected directory layout:
        # feat_root/{train,val,test}
        # mae_feat_root/{train,val,test}
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode

        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")

        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}")

        if not self.how2sign_root.exists():
            raise FileNotFoundError(f"how2sign_root not found: {self.how2sign_root}")

    def _feat_id_from_item(self, item: Dict[str, Any]) -> str:
        """
        Build feature id exactly matching extracted feature filenames.
        """
        sentence_name = item.get("sentence_name")
        if not sentence_name:
            sentence_name = Path(item["video_relpath"]).stem

        start = item.get("start_realigned", item.get("start"))
        if start is None:
            return sentence_name

        start_str = f"{float(start):.2f}".rstrip("0").rstrip(".")
        return f"{sentence_name}_{start_str}"


    def _resolve_feature_path(self, base_dir: Path, feat_id: str, postfix: str) -> Path:
        """Resolve feature path robustly against float formatting mismatches.

        Tries:
          1) exact: <feat_id><postfix>.npy
          2) fallback glob: <feat_id>*<postfix>.npy  (must match exactly 1 file)
        """
        exact = base_dir / f"{feat_id}{postfix}.npy"
        if exact.exists():
            return exact

        # Fallback: tolerate 39 vs 39.0 vs 39.00 etc.
        pattern = str(base_dir / f"{feat_id}*{postfix}.npy")
        hits = glob.glob(pattern)
        if len(hits) == 1:
            return Path(hits[0])

        # Helpful error
        msg = f"Feature not found. Tried exact: {exact}"
        if len(hits) == 0:
            msg += f" | fallback glob found 0 matches: {pattern}"
        else:
            msg += f" | fallback glob matched multiple files ({len(hits)}): " + ", ".join(hits[:5])
        raise FileNotFoundError(msg)

    def _load_spatial_features(self, feat_id: str) -> torch.Tensor:
        path = self._resolve_feature_path(self.spatial_dir, feat_id, self.spatial_postfix)
        return torch.tensor(np.load(path))

    def _load_spatiotemporal_features(self, feat_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(self.spatiotemporal_postfix, str):
            path = self._resolve_feature_path(self.spatiotemporal_dir, feat_id, self.spatiotemporal_postfix)
            return torch.tensor(np.load(path))
        else:
            feats = []
            for postfix in self.spatiotemporal_postfix:
                path = self.spatiotemporal_dir / f"{feat_id}{postfix}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
                feats.append(torch.tensor(np.load(path)))
            return feats

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip()
        if text and not text.endswith("."):
            text += "."
        return text

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # SKIP_MISSING_FEATURES_PATCH_V2
        # If some feature files are missing on disk, skip the sample instead of crashing.
        max_tries = 50
        orig_index = index
        for attempt in range(max_tries):
            try:
                item = self.items[index]
                feat_id = self._feat_id_from_item(item)
                
                pixel_value = self._load_spatial_features(feat_id) if self.spatial else None
                glor_value = self._load_spatiotemporal_features(feat_id) if self.spatiotemporal else None
                
                video_path = self.how2sign_root / item["video_relpath"]
                
                return {
                    "pixel_value": pixel_value,
                    "glor_value": glor_value,
                    "bool_mask_pos": None,
                    "text": self._normalize_text(item.get("sentence", "")),
                    "gloss": "",
                    "id": feat_id,
                    "num_frames": len(pixel_value) if pixel_value is not None else 0,
                    "vid_path": str(video_path),
                    "lang": self.lang,
                    "start": item.get("start_realigned", item.get("start")),
                    "end": item.get("end_realigned", item.get("end")),
                    "original_info": item,
                }
                
            except FileNotFoundError as e:
                # log minimale per evitare spam
                if (attempt < 3) or (attempt % 10 == 0):
                    print(f"[WARN][How2Sign] missing feature -> skip index={{index}}. {e}")
                index = (index + 1) % len(self)
                continue
        raise RuntimeError(f"Too many missing feature files while fetching data. orig_index={{orig_index}} last_index={{index}}")
    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
