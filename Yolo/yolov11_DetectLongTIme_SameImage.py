from ultralytics import YOLO

model = YOLO("yolo11x.pt")

#detect if same person can be detected on same plan.
results = model.track(source="https://www.youtube.com/watch?v=zW-ELjJvXJw", stream=True, persist=True, show=True, save=True)

for r in results:
    print(r.boxes.id)