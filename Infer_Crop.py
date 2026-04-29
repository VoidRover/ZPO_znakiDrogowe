import os
import argparse
import cv2
from ultralytics import YOLO
import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def random_color_for_class(cid):
    np.random.seed(cid)
    color = tuple(int(x) for x in (np.random.randint(50, 230, size=3)))
    return (int(color[0]), int(color[1]), int(color[2]))  # BGR

def draw_label_on_box(img, box, text, color):
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    tx1 = x1 + 2
    ty1 = max(0, y1 + 2)
    tx2 = tx1 + tw + pad
    ty2 = ty1 + th + pad//2
    h, w = img.shape[:2]
    tx2 = min(tx2, w-1)
    ty2 = min(ty2, h-1)
    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1)
    text_x = tx1 + 3
    text_y = ty2 - baseline - 2
    cv2.putText(img, text, (text_x, text_y), font, scale, (255,255,255), thickness, lineType=cv2.LINE_AA)

def predict_and_crop(model_path, image_path, out_dir="./inference_output", conf_thresh=0.25, iou_thresh=0.45):
    if not os.path.exists(model_path):
        print("[ERR] Nie znaleziono modelu:", model_path); return
    if not os.path.exists(image_path):
        print("[ERR] Nie znaleziono obrazu:", image_path); return

    ensure_dir(out_dir)
    model = YOLO(model_path)

    results = model(image_path, conf=conf_thresh, iou=iou_thresh)[0]
    img = cv2.imread(image_path)
    if img is None:
        print("[ERR] nie można wczytać obrazu"); return
    h, w = img.shape[:2]

    if results.boxes is None or len(results.boxes) == 0:
        print("[INFO] Nie wykryto żadnych obiektów.")
        return

    crop_count = 0
    detected_labels = []
    for i, box in enumerate(results.boxes):
        xyxy = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
        detected_labels.append(f"{i} - {label}")
        # Wycinanie
        x1c = max(0, x1); y1c = max(0, y1); x2c = min(w-1, x2); y2c = min(h-1, y2)
        crop = img[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue
        crop_path = os.path.join(out_dir, f"{PathName(image_path).stem}_crop_{i}.jpg")
        cv2.imwrite(crop_path, crop)
        crop_count += 1

        # Rysuj ramkę i etykietę
        color = random_color_for_class(cls_id)
        cv2.rectangle(img, (x1c, y1c), (x2c, y2c), color, 3)
        text = f"{i} - {conf:.2f}"
        draw_label_on_box(img, (x1c, y1c, x2c, y2c), text, color)

        print(f"[INFO] Wykryto: {label} (conf={conf:.2f}) -> crop zapisany: {crop_path}")

    out_img = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_result.jpg")
    cv2.imwrite(out_img, img)
    print(f"[INFO] Zapisano obraz wyniku: {out_img}, liczba cropów: {crop_count}")

    txt_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_labels.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        for line in detected_labels:
            f.write(line + "\n")

    print(f"[INFO] Zapisano listę znaków: {txt_path}")

def PathName(p):
    from pathlib import Path
    return Path(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ścieżka do modelu YOLO (best.pt)")
    parser.add_argument("--image", required=True, help="ścieżka do obrazu")
    parser.add_argument("--out", default="./inference_output", help="katalog na wyniki")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()
    predict_and_crop(args.model, args.image, args.out, args.conf, args.iou)
