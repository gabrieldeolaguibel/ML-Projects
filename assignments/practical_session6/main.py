from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.yaml") # build from scratch using the nano model

# train the model
results = model.train(data="config.yaml", epochs=100) # This automatically does training and validation based on the config.yaml file
# you can view the training validation results in the runs/detect/train folder and see the 'val_batch0_pred.jpg' file to see the results of the model on the validation set
