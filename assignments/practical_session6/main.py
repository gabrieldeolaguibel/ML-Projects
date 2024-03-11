from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.yaml") # build from scratch using the nano model

# train the model
results = model.train(data="config.yaml", epochs=100)