from ultralytics import YOLO

model = YOLO("yolo11n.pt")

#detect on discussion with separated plan for two person.
results = model.track(source="https://www.youtube.com/watch?v=I2ZK3ngNvvI", stream=True, persist=True, show=True, save=True)

for r in results:
    print(r.boxes.id)