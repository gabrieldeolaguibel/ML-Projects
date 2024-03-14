from ultralytics import YOLO
import os
import cv2

''''
Traning the model. 
Results will be in the runs/detect/train folder.
'''

# Load the model
model = YOLO("yolov8n.yaml") # build from scratch using the nano model

# train the model
results = model.train(data="config.yaml", epochs=100) # This automatically does training and validation based on the config.yaml file
# you can view the training validation results in the runs/detect/train folder and see the 'val_batch0_pred.jpg' file to see the results of the model on the validation set

''''
Testing the model on one validation image.
Note: the code above already tests the model on the validation set. This is just an example of how to test the model on a single image since the prof asked for it.
Full validation results are in the runs/detect/train folder/val_batch0_pred.jpg
'''

# Path to image I wante test as validation. Professor will need to change this if he wants to test on his local machine
IMAGE_PATH = '/Users/gabrieldeolaguibel/IE/DevOps_Assignement1/Statistical-Learning-and-Prediction/assignments/practical_session6/data/images/val/d1451008-8d37-441c-a936-242126910691.jpg' # Second to last image in the validation set

# Path to your custom trained YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load the YOLO model
model = YOLO(model_path)  # Load my custom model

# Set the detection threshold
threshold = 0.1 # means we are only interested in objects detected with a confidence of 10% or more

# Load and process the image
image = cv2.imread(IMAGE_PATH)
results = model(image)[0]

# Draw bounding boxes and labels on the image
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Display the image
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows() # Close the window when a key is pressed