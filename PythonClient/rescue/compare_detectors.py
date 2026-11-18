"""Run the .pt model with Ultralytics and the ONNX pipeline on the same image.

    python compare_detectors.py --image output/Victim_01_<time>_raw.png --pt models/person_yolo11s.pt

Needs ultralytics, onnxruntime and opencv (the environment used for
export_onnx.py has them). Prints the people each one finds and saves
<image>_ultralytics.png, <image>_onnx_1tile.png and <image>_onnx_tiled.png
next to the input. If Ultralytics finds people that the 1-tile ONNX run
misses, the ONNX pipeline is wrong; if neither finds them, it's the model.
"""
import argparse
from pathlib import Path

import cv2

from detector import Detection, PersonDetector, draw
from survey_plan import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pt", required=True, help="the original Ultralytics .pt model")
    parser.add_argument("--conf", type=float, help="confidence threshold for all runs (default: config)")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"can't read {args.image}")
    det_cfg = load_config(DEFAULT_CONFIG)["detector"]
    if args.conf is not None:
        det_cfg["conf_threshold"] = args.conf
    stem = Path(args.image).with_suffix("")

    runs = {
        "ultralytics": ultralytics_detect(args.pt, args.image, det_cfg),
        "onnx_1tile": PersonDetector(dict(det_cfg, tiles_x=1, tiles_y=1)).detect(image),
        "onnx_tiled": PersonDetector(det_cfg).detect(image),
    }
    for name, dets in runs.items():
        print(f"{name}: {len(dets)} person(s)")
        for d in dets:
            print(f"  conf {d.confidence:.2f}  {d.width:.0f}x{d.height:.0f} px  at ({d.center[0]:.0f}, {d.center[1]:.0f})")
        out = f"{stem}_{name}.png"
        cv2.imwrite(out, draw(image, dets))
    print(f"saved {stem}_*.png")


def ultralytics_detect(pt_path, image_path, det_cfg):
    from ultralytics import YOLO

    result = YOLO(pt_path).predict(image_path, imgsz=det_cfg["input_size"], conf=det_cfg["conf_threshold"],
                                   iou=det_cfg["iou_threshold"], classes=[det_cfg["person_class"]], verbose=False)[0]
    return [Detection(*box, conf, cls) for box, conf, cls in
            zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.int().tolist())]


if __name__ == "__main__":
    main()
