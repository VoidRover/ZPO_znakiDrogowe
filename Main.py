import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

"""
Użycie:
1. Przygotowanie danych "Pascal VOC" na trening
python Main.py --voc_dir "./sign_pics" --out_dir "./yolo_dataset"
2. Trening na podstawie przygotowanych danych
python Train.py --data ./yolo_dataset/data.yaml --model yolov8n.pt --epochs 20 --imgsz 640 --batch 16
yolo train model=yolov8n.pt data=yolo_dataset/data.yaml imgsz=640 batch=16 epochs=20
3. Użycie wytrenowanego systemu na wybranym obrazie
python Infer_Crop.py --model runs/detect/train2/weights/best.pt --image ./sign.jpg --out ./inference_output
yolo predict model=runs/detect/train2/weights/best.pt source="obraz.jpg" save=True
"""
def voc_to_yolo_bbox(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    return x_center, y_center, w, h

def convert_dataset(voc_dir, out_dir):
    voc_dir = Path(voc_dir)
    out_dir = Path(out_dir)
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"

    for split in ["train", "val", "test"]:
        (out_images / split).mkdir(parents=True, exist_ok=True)
        (out_labels / split).mkdir(parents=True, exist_ok=True)

    xml_files = list(voc_dir.glob("*.xml"))
    images = [f.with_suffix(".jpg") for f in xml_files if f.with_suffix(".jpg").exists()]

    # Split dataset
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.1, random_state=42)

    splits = [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs),
    ]

    # Zbieranie nazw klas
    classes = set()
    for xml in xml_files:
        tree = ET.parse(xml)
        root = tree.getroot()
        for obj in root.findall("object"):
            classes.add(obj.find("name").text)

    classes = sorted(list(classes))
    class_to_id = {cls: i for i, cls in enumerate(classes)}

    # Zapisanie listy klas
    with open(out_dir / "classes.txt", "w") as f:
        for cls in classes:
            f.write(cls + "\n")

    # Konwersja
    print("[INFO] Converting VOC → YOLO")
    for split_name, split_imgs in splits:
        for img_path in tqdm(split_imgs):
            xml_path = img_path.with_suffix(".xml")

            # Skopiuj obraz
            out_img = out_images / split_name / img_path.name
            out_img.write_bytes(img_path.read_bytes())

            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)

            label_file = out_labels / split_name / (img_path.stem + ".txt")

            with open(label_file, "w") as lf:
                for obj in root.findall("object"):
                    cls = obj.find("name").text
                    cls_id = class_to_id[cls]

                    bbox = obj.find("bndbox")
                    xmin = int(float(bbox.find("xmin").text))
                    ymin = int(float(bbox.find("ymin").text))
                    xmax = int(float(bbox.find("xmax").text))
                    ymax = int(float(bbox.find("ymax").text))

                    x, y, w, h = voc_to_yolo_bbox((xmin, ymin, xmax, ymax), img_w, img_h)
                    lf.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    data_yaml = out_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")

    print("[INFO] data.yaml zapisany w:", data_yaml)
    print(f"[INFO] Zakończono konwersję. Dataset YOLO znajduje się w: {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    convert_dataset(args.voc_dir, args.out_dir)

if __name__ == "__main__":
    main()
