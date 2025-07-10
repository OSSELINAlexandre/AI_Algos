from ultralytics import YOLO


model = YOLO("yolo11n.pt")

#Training a model on my own label dataset
results = model.train(data="data/data.yaml", epochs=100, imgsz=640)
