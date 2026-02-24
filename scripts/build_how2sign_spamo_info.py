import json
from pathlib import Path

# === INPUTS (cluster) ===
HOW2SIGN_ROOT = Path("/work/tesi_ztesta/How2Sign")
HOW2SIGN_JSON = HOW2SIGN_ROOT / "how2sign_dataset.json"

# === OUTPUT (inside SpaMo repo) ===
OUT_PATH = Path("dataset/how2sign_spamo_info.json")

# If True, drop samples whose video file doesn't exist (safe default)
DROP_MISSING = True


def main():
    assert HOW2SIGN_JSON.exists(), f"Missing: {HOW2SIGN_JSON}"

    with open(HOW2SIGN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "splits" not in data:
        raise KeyError("how2sign_dataset.json must contain top-level key 'splits'")

    splits = data["splits"]
    for k in ["train", "val", "test"]:
        if k not in splits:
            raise KeyError(f"Missing split '{k}'. Found keys: {list(splits.keys())}")

    out = {
        "meta": data.get("meta", {}),
        "how2sign_root": str(HOW2SIGN_ROOT),  # useful for loaders
        "splits": {"train": [], "val": [], "test": []},
    }

    missing = []

    # In your json, video_path looks like: "data/How2Sign/train_raw_videos/xxx.mp4"
    # On this cluster, videos are in: /work/tesi_ztesta/How2Sign/train_raw_videos/xxx.mp4
    # So we map by keeping only the last two parts: "<split_raw_videos>/<file>.mp4"
    def map_video_path(item):
        rel = Path(item["video_path"])
        parts = rel.parts

        # Find ".../(train_raw_videos|val_raw_videos|test_raw_videos)/file.mp4"
        idx = None
        for i, p in enumerate(parts):
            if p in ("train_raw_videos", "val_raw_videos", "test_raw_videos"):
                idx = i
                break
        if idx is None:
            # fallback: just use filename and hope it's in root (unlikely)
            mapped = HOW2SIGN_ROOT / rel.name
        else:
            mapped = HOW2SIGN_ROOT / Path(*parts[idx:])  # e.g. How2Sign/train_raw_videos/file.mp4
        return mapped

    for split_name in ["train", "val", "test"]:
        for item in splits[split_name]:
            if "video_path" not in item:
                continue

            video_abs = map_video_path(item)

            if not video_abs.exists():
                missing.append(str(video_abs))
                if DROP_MISSING:
                    continue

            out_item = {
                "split": split_name,
                # We save RELATIVE path w.r.t HOW2SIGN_ROOT -> portable on cluster
                "video_relpath": str(video_abs.relative_to(HOW2SIGN_ROOT)),
                "sentence": item.get("sentence", ""),
                "start": item.get("start_realigned"),
                "end": item.get("end_realigned"),
                # extras (nice to have)
                "video_id": item.get("video_id"),
                "sentence_id": item.get("sentence_id"),
                "video_name": item.get("video_name"),
                "sentence_name": item.get("sentence_name"),
            }
            out["splits"][split_name].append(out_item)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {OUT_PATH.resolve()}")
    print("Counts:", {k: len(v) for k, v in out["splits"].items()})

    if missing:
        miss_path = OUT_PATH.with_suffix(".missing.txt")
        with open(miss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        print(f"WARNING: {len(missing)} missing videos. List saved to: {miss_path.resolve()}")


if __name__ == "__main__":
    main()
