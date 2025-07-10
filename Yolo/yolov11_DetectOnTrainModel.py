from ultralytics import YOLO


#best_pt is a train model.
model = YOLO("runs/detect/train5/weights/best.pt")
results = model('test.jpg')

results[0].show()