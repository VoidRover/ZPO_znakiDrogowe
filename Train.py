from ultralytics import YOLO
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="ścieżka do data.yaml (np. yolo_dataset/data.yaml)")
    p.add_argument("--model", default="yolov8n.pt", help="początkowy model (np. yolov8n.pt) lub 'yolov8n.pt'")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()
    print("[INFO] Trening YOLO z:", args.data)
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    print("[INFO] Trenowanie zakończone. Szukaj wyników w runs/")

if __name__ == "__main__":
    main()
