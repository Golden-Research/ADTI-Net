#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert raw object-detection annotations into the unified COCO-VID style
JSON that this repository consumes (see datasets/vid_multi.py / vid_single.py).

Supported source formats
------------------------
1) ImageNet VID  (annotation = one XML per frame, VOC-like layout)
     <ann_root>/<...>/<video>/000000.xml
   * the <name> field inside each <object> is an ImageNet wnid, mapped to
     category id 1..30 with the same table used by mmtracking
     (CLASSES / CLASSES_ENCODES below).
2) UAVDT (one *_gt_whole.txt per video, 9 CSV columns per object)
     --ann-dir <UAVDT>/UAV-benchmark-MOTD_v1.0/GT
     --img-root <UAVDT>/UAV-benchmark-M
   Frame ID, target ID, left, top, width, height, out-of-view,
   occlusion, category (1 car, 2 truck, 3 bus).
   Use --split train/test with official M_attr/{train,test} sequence lists.

Both are written to the SAME JSON schema (videos/images/annotations/
categories) so you can train/evaluate either dataset with this repo.

Usage examples
--------------
# ImageNet VID
python tools/convert_to_vid_json.py --dataset imagenet_vid \
    --ann-dir <ILSVRC2015>/Annotations/VID/val \
    --img-root <ILSVRC2015>/Data/VID/val \
    --out annotations/imagenet_vid_val.json

python tools/convert_to_vid_json.py --dataset imagenet_vid \
    --ann-dir <ILSVRC2015>/Annotations/VID/train \
    --img-root <ILSVRC2015>/Data/VID/train \
    --out annotations/imagenet_vid_train.json

# UAVDT: official train/test split identified by M_attr files
python tools/convert_to_vid_json.py --dataset uavdt \
    --ann-dir datasets/uavdt/UAV-benchmark-MOTD_v1.0/GT \
    --img-root datasets/uavdt/UAV-benchmark-M \
    --split train --split-dir datasets/uavdt/M_attr \
    --out annotations/uavdt_train.json

python tools/convert_to_vid_json.py --dataset uavdt \
    --ann-dir datasets/uavdt/UAV-benchmark-MOTD_v1.0/GT \
    --img-root datasets/uavdt/UAV-benchmark-M \
    --split test --split-dir datasets/uavdt/M_attr \
    --out annotations/uavdt_test.json

Note on the file_name field: file_name is written relative to --img-root.
Point the img_folder of the dataset you build (datasets/vid_*.py PATHS) at
that --img-root directory (or symlink it there) so images resolve correctly.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

try:
    import xml.etree.cElementTree as ET
except ImportError:  # pragma: no cover
    import xml.etree.ElementTree as ET

logger = logging.getLogger("convert_to_vid_json")


# --------------------------------------------------------------------------- #
# ImageNet VID: 30 classes, order (and wnids) identical to mmtracking so the
# produced json is interchangeable with the official converted files.
# --------------------------------------------------------------------------- #
VID_CLASSES = (
    "airplane", "antelope", "bear", "bicycle", "bird",
    "bus", "car", "cattle", "dog", "domestic_cat",
    "elephant", "fox", "giant_panda", "hamster", "horse",
    "lion", "lizard", "monkey", "motorcycle", "rabbit",
    "red_panda", "sheep", "snake", "squirrel", "tiger",
    "train", "turtle", "watercraft", "whale", "zebra",
)
VID_CLASSES_ENCODES = (
    "n02691156", "n02419796", "n02131653", "n02834778", "n01503061",
    "n02924116", "n02958343", "n02402425", "n02084071", "n02121808",
    "n02503517", "n02118333", "n02510455", "n02342885", "n02374451",
    "n02129165", "n01674464", "n02484322", "n03790512", "n02324045",
    "n02509815", "n02411705", "n01726692", "n02355227", "n02129604",
    "n04468005", "n01662784", "n04530566", "n02062744", "n02391049",
)
VID_NAME_TO_ID = {wnid: i for i, wnid in enumerate(VID_CLASSES_ENCODES, 1)}

# --------------------------------------------------------------------------- #
# UAVDT detection classes in *_gt_whole.txt
UAVDT_NAMES = {1: "car", 2: "truck", 3: "bus"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def natural_key(path: Path):
    m = re.search(r"\d+", path.stem)
    return (int(m.group()), path.name) if m else (10 ** 12, path.name)


def find_image(img_root: Path, ann_rel_dir: str, stem: str):
    """Locate the image that matches an annotation file and return its posix
    path relative to img_root, or None if not found."""
    for ext in ("jpg", "jpeg", "png"):
        for name in (f"{stem}.{ext}", f"{stem}.{ext.upper()}"):
            p = img_root / ann_rel_dir / name
            if p.is_file():
                return p.relative_to(img_root).as_posix()
    return None


def read_image_size(img_path: Path):
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# group discovery: directories that directly contain the annotation files
# --------------------------------------------------------------------------- #
def discover_annotation_groups(ann_dir: str, suffix: str):
    """Return a list of (video_name, rel_dir, [frame_path, ...]) sorted.

    - Nested layout (e.g. ImageNet VID train: .../<video>/*.xml or
      VisDrone VID: annotations/<video>/*.txt) -> each folder holding files
      is one video.
    - Flat layout (e.g. VisDrone DET: annotations/*.txt) -> every file is
      treated as a single-frame video so temporal neighbours are never
      sampled across unrelated images.
    """
    root = Path(ann_dir)
    assert root.is_dir(), f"annotation dir does not exist: {root}"
    groups = []
    for dirpath, _dirnames, filenames in os.walk(root):
        matched = sorted(
            (Path(dirpath) / f for f in filenames
             if f.lower().endswith(suffix)),
            key=natural_key)
        if not matched:
            continue
        rel = Path(dirpath).relative_to(root)
        if rel == Path("."):
            # flat: one file per (single-frame) video
            for fp in matched:
                groups.append((fp.stem, "", [fp]))
        else:
            groups.append((rel.as_posix(), rel.as_posix(), matched))
    groups.sort(key=lambda g: g[0])
    return groups


# --------------------------------------------------------------------------- #
# ImageNet VID XML parsing
# --------------------------------------------------------------------------- #
def parse_imagenet_video(xml_paths, ann_rel_dir, img_root, vid_id, counters):
    images, anns = [], []
    # trackid -> global instance id, valid within this video only
    track_map = {}
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()

        size = root.find("size")
        try:
            width = int(size.find("width").text)
            height = int(size.find("height").text)
        except Exception:
            width = height = 0

        stem = xml_path.stem
        file_name = None
        if img_root is not None:
            file_name = find_image(img_root, ann_rel_dir, stem)
        if file_name is None:
            file_name = f"{ann_rel_dir}/{stem}.JPEG" if ann_rel_dir else f"{stem}.JPEG"
            logger.warning("image not found for %s, using %s", xml_path,
                           file_name)

        counters["img_id"] += 1
        img_id = counters["img_id"]
        images.append(dict(
            id=img_id, file_name=file_name, width=width, height=height,
            video_id=vid_id, frame_id=len(images),
        ))

        for obj in root.findall("object"):
            name_el = obj.find("name")
            name = name_el.text if name_el is not None else ""
            if name not in VID_NAME_TO_ID:
                continue
            bndbox = obj.find("bndbox")
            try:
                x1 = float(bndbox.find("xmin").text)
                y1 = float(bndbox.find("ymin").text)
                x2 = float(bndbox.find("xmax").text)
                y2 = float(bndbox.find("ymax").text)
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            counters["ann_id"] += 1
            ann_id = counters["ann_id"]
            track_el = obj.find("trackid")
            if track_el is not None and track_el.text is not None:
                trackid = track_el.text
                if trackid not in track_map:
                    counters["instance_id"] += 1
                    track_map[trackid] = counters["instance_id"]
                instance_id = track_map[trackid]
            else:
                instance_id = ann_id

            occluded = obj.find("occluded")
            generated = obj.find("generated")
            w = x2 - x1
            h = y2 - y1
            anns.append(dict(
                id=ann_id, video_id=vid_id, image_id=img_id,
                category_id=VID_NAME_TO_ID[name], instance_id=instance_id,
                bbox=[x1, y1, w, h], area=w * h, iscrowd=False,
                occluded=bool(occluded is not None and occluded.text == "1"),
                generated=bool(generated is not None and generated.text == "1"),
            ))
    return images, anns


# --------------------------------------------------------------------------- #
# UAVDT uses ONE annotation file per VIDEO, not one TXT file per frame.
# *_gt_whole.txt fields:
# frame_index,target_id,left,top,width,height,out_of_view,occlusion,category
# --------------------------------------------------------------------------- #
def convert_uavdt(args):
    if args.img_root is None:
        raise ValueError("--img-root is required for uavdt")
    img_root = Path(args.img_root)
    ann_root = Path(args.ann_dir)
    if not img_root.is_dir() or not ann_root.is_dir():
        raise FileNotFoundError("UAVDT image or annotation directory not found")
    if args.split != "all":
        if not args.split_dir:
            raise ValueError("--split-dir is required for --split train/test")
        split_root = Path(args.split_dir) / args.split
        if not split_root.is_dir():
            raise FileNotFoundError(f"UAVDT split directory not found: {split_root}")
        seq_names = sorted({f.name.removesuffix("_attr.txt")
                            for f in split_root.glob("*_attr.txt")})
        if not seq_names:
            raise ValueError(f"No *_attr.txt files found in {split_root}")
    else:
        seq_names = sorted(p.name.removesuffix("_gt_whole.txt")
                           for p in ann_root.glob("*_gt_whole.txt"))
    if not seq_names:
        raise ValueError("No UAVDT *_gt_whole.txt annotations found")
    videos, images, anns = [], [], []
    counters = dict(img_id=0, ann_id=0, instance_id=0)
    for name in seq_names:
        seq_dir = img_root / name
        gt_path = ann_root / f"{name}_gt_whole.txt"
        if not seq_dir.is_dir() or not gt_path.is_file():
            raise FileNotFoundError(f"Missing UAVDT sequence images or GT: {seq_dir}, {gt_path}")
        frame_paths = sorted((p for p in seq_dir.iterdir()
                              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                             key=natural_key)
        if not frame_paths:
            raise ValueError(f"No frames in {seq_dir}")
        video_id = len(videos) + 1
        videos.append(dict(id=video_id, name=name))
        frame_images = {}
        for pos, path in enumerate(frame_paths):
            if pos % args.frame_step:
                continue
            matched = re.fullmatch(r"img(\d+)", path.stem, flags=re.I)
            if not matched:
                raise ValueError(f"Expected UAVDT frame name img000001.jpg, got {path}")
            frame_no = int(matched.group(1))
            if frame_no in frame_images:
                raise ValueError(f"Duplicate frame index {frame_no} in {seq_dir}")
            wh = read_image_size(path)
            if wh is None:
                raise ValueError(f"Cannot read image size: {path}; install Pillow")
            counters["img_id"] += 1
            image_id = counters["img_id"]
            images.append(dict(id=image_id, file_name=path.relative_to(img_root).as_posix(),
                               width=wh[0], height=wh[1], video_id=video_id,
                               frame_id=pos))
            frame_images[frame_no] = (image_id, wh)
        track_map = {}
        with gt_path.open(encoding="utf-8-sig") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                cols = [v.strip() for v in line.split(",")]
                if len(cols) != 9:
                    raise ValueError(f"{gt_path}:{line_no}: expected 9 fields, got {len(cols)}")
                try:
                    frame_no, target_id = int(cols[0]), int(cols[1])
                    x, y, w, h = map(float, cols[2:6])
                    out_of_view, occlusion, category = map(int, cols[6:9])
                except ValueError as exc:
                    raise ValueError(f"{gt_path}:{line_no}: invalid UAVDT ground truth") from exc
                if category not in UAVDT_NAMES:
                    raise ValueError(f"{gt_path}:{line_no}: unknown category {category}")
                if frame_no not in frame_images or w <= 0 or h <= 0:
                    continue
                image_id, (width, height) = frame_images[frame_no]
                # Clip to image bounds; skip boxes wholly outside the image.
                x2, y2 = min(float(width), x + w), min(float(height), y + h)
                x1, y1 = max(0.0, x), max(0.0, y)
                if x2 <= x1 or y2 <= y1:
                    continue
                if target_id not in track_map:
                    counters["instance_id"] += 1
                    track_map[target_id] = counters["instance_id"]
                counters["ann_id"] += 1
                anns.append(dict(id=counters["ann_id"], video_id=video_id,
                                 image_id=image_id, category_id=category,
                                 instance_id=track_map[target_id],
                                 bbox=[x1, y1, x2-x1, y2-y1],
                                 area=(x2-x1)*(y2-y1), iscrowd=False,
                                 out_of_view=out_of_view, occlusion=occlusion))
        logger.info("UAVDT sequence %s: %d retained frames", name, len(frame_images))
    return dict(videos=videos, images=images, annotations=anns,
                categories=[dict(id=k, name=v) for k, v in UAVDT_NAMES.items()])


# --------------------------------------------------------------------------- #
# top-level converters
# --------------------------------------------------------------------------- #
def convert_imagenet_vid(args):
    if args.img_root is None:
        logger.error("--img-root is required for imagenet_vid")
        sys.exit(1)
    groups = discover_annotation_groups(args.ann_dir, ".xml")
    if not groups:
        logger.error("no .xml annotation found under %s", args.ann_dir)
        sys.exit(1)
    logger.info("%d videos found", len(groups))

    videos, images, anns = [], [], []
    counters = dict(img_id=0, ann_id=0, instance_id=0)
    img_root = Path(args.img_root)
    for video_name, rel_dir, xml_paths in groups:
        kept = xml_paths[:: args.frame_step]
        vid_id = len(videos) + 1
        videos.append(dict(id=vid_id, name=video_name))
        im, an = parse_imagenet_video(kept, rel_dir, img_root, vid_id, counters)
        images.extend(im)
        anns.extend(an)
        if len(videos) % 500 == 0:
            logger.info("... %d videos / %d images", len(videos), len(images))

    categories = [
        dict(id=i, name=name, encode_name=wnid)
        for i, (name, wnid) in enumerate(
            zip(VID_CLASSES, VID_CLASSES_ENCODES), 1)
    ]
    return dict(videos=videos, images=images, annotations=anns,
                categories=categories)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Convert ImageNet VID (XML) / UAVDT (video-level TXT) annotations "
                    "into the unified COCO-VID JSON used by this repo.")
    parser.add_argument("--dataset", choices=["imagenet_vid", "uavdt"],
                        required=True, help="source annotation format")
    parser.add_argument("--ann-dir", required=True,
                        help="root dir that contains the annotation files "
                             "(XML for imagenet_vid, *_gt_whole.txt for uavdt)")
    parser.add_argument("--img-root", default=None,
                        help="root dir of the images; file_name in the output "
                             "json is written relative to this dir")
    parser.add_argument("--out", required=True,
                        help="output json path (e.g. annotations/xxx.json)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="keep every Nth frame of each video (default 1 = "
                             "all frames). Larger values subsample, e.g. 15.")
    parser.add_argument("--split", choices=["train", "test", "all"],
                        default="all", help="UAVDT official train/test split; all uses every GT")
    parser.add_argument("--split-dir", default=None,
                        help="UAVDT M_attr directory containing train/ and test/ sequence lists")
    parser.add_argument("--indent", type=int, default=None,
                        help="json indent (omit for compact output)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)

    if args.frame_step < 1:
        parser.error("--frame-step must be >= 1")

    convert = (convert_imagenet_vid if args.dataset == "imagenet_vid"
               else convert_uavdt)
    out_data = convert(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, indent=args.indent)

    logger.info("done: %s", out_path)
    logger.info("  videos=%d images=%d annotations=%d categories=%d",
                len(out_data["videos"]), len(out_data["images"]),
                len(out_data["annotations"]), len(out_data["categories"]))
    # sanity: every image / annotation knows its video
    assert all("video_id" in im for im in out_data["images"])
    assert all("video_id" in an for an in out_data["annotations"])


if __name__ == "__main__":
    main()
