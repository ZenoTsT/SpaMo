import json
from pathlib import Path

MANIFEST = Path("dataset/how2sign_spamo_info.json")
OUT_DIR = Path("preprocess/How2Sign")

# Columns: fileid \t video_relpath \t text
# fileid = stem del video (senza .mp4)
def write_tsv(items, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("fileid\tvideo\ttext\n")
        for it in items:
            rel = it["video_relpath"]
            fileid = Path(rel).stem
            text = (it.get("sentence") or "").replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{fileid}\t{rel}\t{text}\n")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = json.load(open(MANIFEST, "r", encoding="utf-8"))
    splits = d["splits"]

    write_tsv(splits["train"], OUT_DIR / "train.tsv")
    # mappiamo val -> dev
    write_tsv(splits["val"], OUT_DIR / "dev.tsv")
    write_tsv(splits["test"], OUT_DIR / "test.tsv")

    print("Wrote:")
    for p in ["train.tsv", "dev.tsv", "test.tsv"]:
        fp = OUT_DIR / p
        print(fp, "lines:", sum(1 for _ in open(fp, "r", encoding="utf-8")) - 1)

if __name__ == "__main__":
    main()
