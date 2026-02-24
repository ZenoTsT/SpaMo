import json
import numpy as np
from pathlib import Path

MANIFEST = Path("dataset/how2sign_spamo_info.json")
OUT_DIR = Path("preprocess/How2Sign")

def to_dict_list(split_items):
    out = {}
    for i, it in enumerate(split_items):
        rel = it["video_relpath"]      # train_raw_videos/xxx.mp4
        fileid = Path(rel).stem
        text = (it.get("sentence") or "").strip()
        start = it.get("start")
        end = it.get("end")

        out[i] = {
            "fileid": fileid,
            "text": text,
            "gloss": "",
            "folder": rel,
            "original_info": {          # ✅ richiesto da vit_extract_feature.py
                "START_REALIGNED": start,
                "END_REALIGNED": end,
            },
        }
    return out

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = json.load(open(MANIFEST, "r", encoding="utf-8"))
    splits = d["splits"]

    mapping = {
        "train": "train_info.npy",
        "val": "val_info.npy",
        "test": "test_info.npy",
    }

    for split_name, out_name in mapping.items():
        arr = to_dict_list(splits[split_name])
        np.save(OUT_DIR / out_name, arr)
        print(f"Saved {OUT_DIR/out_name} with {len(arr)} items")

if __name__ == "__main__":
    main()
