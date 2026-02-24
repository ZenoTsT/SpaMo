import os
import os.path as osp
import argparse
import uuid
import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor

import sys
sys.path.append("./")

from utils.helpers import sliding_window_for_list, read_video, get_img_list

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


# ----------------------------
# Robust skip / resume helpers
# ----------------------------
def is_valid_feature_file(path: str) -> bool:
    """
    Valid if it is a .npy containing a non-empty 2D array: (num_windows, hidden_dim).
    We use np.load(mmap_mode="r") to keep it cheap.
    """
    if not osp.exists(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return getattr(arr, "ndim", 0) == 2 and arr.shape[0] > 0 and arr.shape[1] > 0
    except Exception:
        return False


def atomic_save_npy(out_file: str, arr: np.ndarray):
    """
    Atomically save .npy safely.
    IMPORTANT: use a file handle so np.save does NOT append ".npy" unexpectedly.
    """
    out_dir = osp.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    tmp_file = out_file + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(tmp_file, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_file, out_file)
    finally:
        # best-effort cleanup if something went wrong before replace
        if osp.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


# ----------------------------
# Feature reader
# ----------------------------
class VideoMAEFeatureReader(object):
    def __init__(
        self,
        model_name="MCG-NJU/videomae-large",
        cache_dir=None,
        device="cuda:0",
        overlap_size=8,
        nth_layer=-1,
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = VideoMAEModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device).eval()

    @torch.no_grad()
    def get_feats(self, video):
        # video: list of list[PIL.Image] with shape [B][16]
        inputs = self.image_processor(images=video, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        x = outputs[self.nth_layer]         # (B, tokens, hidden)
        x = x[:, 0]                         # CLS token -> (B, hidden)
        return x


# ----------------------------
# CLI
# ----------------------------
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--anno_root", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--save_dir", required=True)

    p.add_argument("--model_name", default="MCG-NJU/videomae-large")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overlap_size", type=int, default=8)
    p.add_argument("--nth_layer", type=int, default=-1)
    p.add_argument("--cache_dir", default=None)

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)

    # optional: limit items per split for debugging
    p.add_argument("--max_items", type=int, default=-1, help="Process at most N items per split (debug)")
    return p


def build_out_file(out_dir: str, fileid: str, st: str, overlap: int) -> str:
    postfix = f"_overlap-{overlap}"
    if st is not None:
        postfix = f"_{st}{postfix}"
    return osp.join(out_dir, f"{fileid}{postfix}.npy")


def shard_total(num_items: int, num_shards: int, shard_id: int) -> int:
    # how many indices i in [0, num_items) satisfy i % num_shards == shard_id
    return (num_items - shard_id + (num_shards - 1)) // num_shards


def iter_dataset(args, mode_key: str, out_dir: str, reader: VideoMAEFeatureReader):
    """
    mode_key: one of {"train","val","test"} to match *_info.npy in preprocess.
    yields (feats_arr, fileid, st_str_or_None, out_file)
    """
    data = np.load(osp.join(args.anno_root, f"{mode_key}_info.npy"), allow_pickle=True).item()
    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]

    # counters
    skip = write = invalid = 0
    processed = 0

    total = shard_total(num, args.num_shards, args.shard_id)

    pbar = tqdm.tqdm(total=total, desc=f"{mode_key} | skip={skip} write={write} invalid={invalid}")

    for i in range(args.shard_id, num, args.num_shards):
        if args.max_items > 0 and processed >= args.max_items:
            break

        fileid = data[i]["fileid"]
        fname = data[i]["folder"]

        st = None
        if ds_name == "How2Sign":
            st = str(data[i]["original_info"]["START_REALIGNED"])

        out_file = build_out_file(out_dir, fileid, st, args.overlap_size)

        # ---- SKIP (resume) before doing any heavy work ----
        if is_valid_feature_file(out_file):
            skip += 1
            pbar.set_description(f"{mode_key} | skip={skip} write={write} invalid={invalid}")
            pbar.update(1)
            processed += 1
            continue
        # -----------------------------------------------

        # load frames
        if ds_name in ["Phoenix14T", "CSL-Daily"]:
            image_list = get_img_list(ds_name, args.video_root, fname)
            if len(image_list) == 0:
                invalid += 1
                pbar.set_description(f"{mode_key} | skip={skip} write={write} invalid={invalid}")
                pbar.update(1)
                processed += 1
                continue

            if len(image_list) < 16:
                image_list.extend([image_list[-1]] * (16 - len(image_list)))

            chunks = sliding_window_for_list(image_list, window_size=16, overlap_size=args.overlap_size)
            videos = [[Image.open(im).convert("RGB") for im in ch] for ch in chunks]
        else:
            # How2Sign / others: fname already includes "train_raw_videos/..mp4" in your preprocess
            video_rel = fname
            if isinstance(video_rel, str) and video_rel.startswith("/"):
                video_path = video_rel
            else:
                if isinstance(video_rel, str) and (
                    "train_raw_videos/" in video_rel or "val_raw_videos/" in video_rel or "test_raw_videos/" in video_rel
                ):
                    video_path = osp.join(args.video_root, video_rel)
                else:
                    split_dir = {"train": "train_raw_videos", "val": "val_raw_videos", "test": "test_raw_videos"}[mode_key]
                    if isinstance(video_rel, str) and video_rel.endswith(".mp4"):
                        video_path = osp.join(args.video_root, split_dir, video_rel)
                    else:
                        video_path = osp.join(args.video_root, split_dir, f"{video_rel}.mp4")

            frames = read_video(video_path)
            if len(frames) == 0:
                invalid += 1
                pbar.set_description(f"{mode_key} | skip={skip} write={write} invalid={invalid}")
                pbar.update(1)
                processed += 1
                continue

            if len(frames) < 16:
                frames.extend([frames[-1]] * (16 - len(frames)))

            videos = sliding_window_for_list(frames, window_size=16, overlap_size=args.overlap_size)

        # compute features in batches
        video_feats = []
        for j in range(0, len(videos), args.batch_size):
            video_batch = videos[j : min(j + args.batch_size, len(videos))]
            feats = reader.get_feats(video_batch).cpu().numpy()
            video_feats.append(feats)

        feats_arr = np.concatenate(video_feats, axis=0) if len(video_feats) > 0 else np.zeros((0, 1), dtype=np.float32)

        # save atomically
        atomic_save_npy(out_file, feats_arr)
        write += 1

        pbar.set_description(f"{mode_key} | skip={skip} write={write} invalid={invalid}")
        pbar.update(1)
        processed += 1

    pbar.close()


def main():
    args = get_parser().parse_args()

    ds_name = osp.split(args.anno_root)[-1]
    fname = f"mae_feat_{ds_name}"

    reader = VideoMAEFeatureReader(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        overlap_size=args.overlap_size,
        nth_layer=args.nth_layer,
    )

    # keep the split folder names expected by your config (dev/test/train),
    # but load the npy keys as (val/test/train) for How2Sign.
    split_map = [("dev", "val"), ("test", "test"), ("train", "train")]

    for out_split, mode_key in split_map:
        out_dir = osp.join(args.save_dir, fname, out_split)
        os.makedirs(out_dir, exist_ok=True)
        iter_dataset(args, mode_key, out_dir, reader)


if __name__ == "__main__":
    main()
