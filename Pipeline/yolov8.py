from ultralytics import YOLO
import cv2


def predict(model_path, image_path):
    # Load the YOLO model
    model = YOLO(model_path)

    # Load the image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]  # Get image dimensions

    # Run inference on the image
    results = model([img])  # Adjusted to pass a list of images

    output_strings = []

    for result in results:
        # Extract bounding boxes (xyxy format), classes, and confidences
        boxes = result.boxes.xyxy.cpu().numpy()  # Convert boxes to numpy array
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        keypoints = result.keypoints.xy.cpu().numpy()  # Convert keypoints to numpy array

        for i, box in enumerate(boxes):
            cls_id = classes[i]
            class_name = result.names[int(cls_id)]
            x1, y1, x2, y2 = box[:4]
            # Normalize coordinates
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Format and normalize keypoints
            keypoints_formatted = []
            for kp in keypoints[i]:  # Assuming correct keypoint-box alignment
                # Normalize keypoint coordinates
                norm_x = kp[0] / img_width
                norm_y = kp[1] / img_height
                kp_str = "{:.4f},{:.4f},{:.2f}".format(norm_x, norm_y, kp[2] if len(kp) > 2 else 0)  # Example formatting
                keypoints_formatted.append(kp_str)

            output_string = f"{class_name},{center_x:.4f},{center_y:.4f},{width:.4f},{height:.4f}," + ",".join(keypoints_formatted)
            output_strings.append(output_string)

    return output_strings
