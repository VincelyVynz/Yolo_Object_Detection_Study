from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model(source=0, show=True)
