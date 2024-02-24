from ultralytics import YOLO
import cv2

model_path = 'C:\\Users\\sakar\\Semester Project\\semester-project-yolov8\\runs\\pose\\train22\\weights\\last.pt'
image_path = 'C:\\Users\\sakar\\Semester Project\\semester-project-yolov8\\spot.jpg'

def predict():
    # Load the YOLO model
    model = YOLO(model_path)

    # Load the image
    img = cv2.imread(image_path)

    # Run inference on the image
    results = model([img])  # Adjusted to pass a list of images

    output_strings = []

    for result in results:
        # Extract bounding boxes (xyxy format), classes, and confidences
        boxes = result.boxes.xyxy.cpu().numpy()  # Convert boxes to numpy array
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Assuming each result corresponds to one image and all keypoints for detected objects are in one array
        keypoints = result.keypoints.xy.cpu().numpy()  # Convert keypoints to numpy array

        for i, box in enumerate(boxes):
            cls_id = classes[i]
            class_name = result.names[int(cls_id)]
            x1, y1, x2, y2 = box[:4]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1

            # Assuming you have a mechanism to align or associate keypoints with boxes, which might require custom logic
            # Here's a simplified placeholder for keypoint formatting assuming they are directly associated
            # Adjust the keypoint extraction logic based on your actual data structure and needs
            keypoints_formatted = []
            for kp in keypoints[i]:  # Adjust this index based on your keypoint-box alignment logic
                # Ensure kp is an individual keypoint with its elements accessible
                kp_str = "{:.2f},{:.2f},{:.2f}".format(kp[0], kp[1], kp[2] if len(kp) > 2 else 0)  # Example formatting
                keypoints_formatted.append(kp_str)

            output_string = f"{class_name},{center_x},{center_y},{width},{height}," + ",".join(keypoints_formatted)
            output_strings.append(output_string)

            print(output_string)

    return output_strings

# Call the predict function
output_strings = predict()
