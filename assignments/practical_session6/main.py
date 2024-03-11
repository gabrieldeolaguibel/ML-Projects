from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.yaml") # build from scratch

# train the model
results = model.train(data="config.yaml", epochs=100)