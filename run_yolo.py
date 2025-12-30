from ultralytics import YOLO

model = YOLO("yolov8l.pt")
results = model("street_1.jpg", save = False)

r = results[0]

print("Classes:", r.names, end="\n\n")
print("Boxes:", r.boxes.xyxy, end="\n\n")
print("Conf:", r.boxes.conf, end="\n\n")
print("Cls:", r.boxes.cls)

results[0].show()