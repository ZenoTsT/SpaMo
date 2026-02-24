import os
import os.path as osp
import argparse
import uuid
import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel

import sys
sys.path.append("./")

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import read_video, get_img_list

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


# ----------------------------
# Robust skip / resume helpers (same spirit as MAE)
# ----------------------------
def is_valid_feature_file(path: str) -> bool:
    """
    Valid if it is a .npy containing a non-empty 2D array: (T, D).
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
        if osp.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


def shard_total(num_items: int, num_shards: int, shard_id: int) -> int:
    return (num_items - shard_id + (num_shards - 1)) // num_shards


def build_out_file(out_dir: str, fileid: str, st: str, s2_mode: str, scales: list) -> str:
    postfix = ""
    if s2_mode:
        postfix = f"_{s2_mode}"
        if len(scales) == 3:
            postfix = f"{postfix}_large"
    if st is not None:
        postfix = f"_{st}{postfix}"
    return osp.join(out_dir, f"{fileid}{postfix}.npy")


# ----------------------------
# Feature reader
# ----------------------------
class ViTFeatureReader(object):
    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        cache_dir=None,
        device="cuda:0",
        s2_mode="",
        scales=None,
        nth_layer=-1,
    ):
        self.device = device
        self.s2_mode = s2_mode or ""
        self.scales = scales or []
        self.nth_layer = nth_layer

        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(self.device).eval()

        self.image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    @torch.no_grad()
    def forward_features(self, inputs):
        # inputs: (B, C, H, W)
        outputs = self.model(inputs).hidden_states
        return outputs[self.nth_layer]  # (B, tokens, D)

    @torch.no_grad()
    def get_feats(self, frames):
        """
        frames: list[PIL.Image] or list[np.ndarray] already converted in helpers
        returns: (T, D) numpy later, where D=1024 (no S²) or 2048 (S² with scales 1 2)
        """
        inputs = self.image_processor(list(frames), return_tensors="pt").to(self.device).pixel_values
        if self.s2_mode == "s2wrapping":
            x = multiscale_forward(self.forward_features, inputs, scales=self.scales, num_prefix_token=1)
        else:
            x = self.forward_features(inputs)
        x = x[:, 0]  # CLS -> (T, D)
        return x


# ----------------------------
# CLI
# ----------------------------
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--anno_root", required=True, help="location of preprocess folder containing *_info.npy")
    p.add_argument("--video_root", required=True, help="root folder of dataset videos/frames")
    p.add_argument("--save_dir", required=True, help="where to save the output")

    p.add_argument("--model_name", default="openai/clip-vit-large-patch14", help="ViT model name")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nth_layer", type=int, default=-1)
    p.add_argument("--cache_dir", default=None)

    p.add_argument("--s2_mode", default="", help='use "s2wrapping" to enable S²')
    p.add_argument("--scales", nargs="+", type=int, default=[], help="List of scales, e.g. 1 2")

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)

    # optional: limit items per split for debugging
    p.add_argument("--max_items", type=int, default=-1, help="Process at most N items per split (debug)")
    return p


def iter_dataset(args, mode_key: str, out_dir: str, reader: ViTFeatureReader):
    """
    mode_key: one of {"train","val","test"} to match *_info.npy in preprocess.
    yields (feats_arr, fileid, st_str_or_None, out_file)
    """
    data = np.load(osp.join(args.anno_root, f"{mode_key}_info.npy"), allow_pickle=True).item()
    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]

    processed = 0
    total = shard_total(num, args.num_shards, args.shard_id)

    skip = write = invalid = 0
    pbar = tqdm.tqdm(total=total, desc=f"{mode_key} | skip={skip} write={write} invalid={invalid}")

    for i in range(args.shard_id, num, args.num_shards):
        if args.max_items > 0 and processed >= args.max_items:
            break

        fileid = data[i]["fileid"]
        fname = data[i]["folder"]

        st = None
        if ds_name == "How2Sign":
            st = str(data[i]["original_info"]["START_REALIGNED"])

        out_file = build_out_file(out_dir, fileid, st, args.s2_mode, args.scales)

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

            frames = [Image.open(im).convert("RGB") for im in image_list]
        else:
            # How2Sign / others: fname may already include "train_raw_videos/..mp4" in your preprocess
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

        # compute features in batches
        feats_list = []
        for j in range(0, len(frames), args.batch_size):
            batch = frames[j : min(j + args.batch_size, len(frames))]
            x = reader.get_feats(batch).cpu().numpy()
            feats_list.append(x)

        feats_arr = np.concatenate(feats_list, axis=0) if len(feats_list) > 0 else np.zeros((0, 1), dtype=np.float32)

        atomic_save_npy(out_file, feats_arr)
        write += 1

        pbar.set_description(f"{mode_key} | skip={skip} write={write} invalid={invalid}")
        pbar.update(1)
        processed += 1

    pbar.close()


def main():
    args = get_parser().parse_args()

    ds_name = osp.split(args.anno_root)[-1]
    _model_name = os.path.split(args.model_name)[-1]
    fname = f"{_model_name}_feat_{ds_name}"

    reader = ViTFeatureReader(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer,
    )

    # Keep the split folder names expected by config (dev/test/train),
    # but load the npy keys as (val/test/train) for How2Sign (same as MAE).
    split_map = [("dev", "val"), ("test", "test"), ("train", "train")]

    for out_split, mode_key in split_map:
        out_dir = osp.join(args.save_dir, fname, out_split)
        os.makedirs(out_dir, exist_ok=True)
        iter_dataset(args, mode_key, out_dir, reader)


if __name__ == "__main__":
    main()
