import simple_yet_effective as sye
import yolov8

image_path = 'C:\\Users\\sakar\\Semester Project\\semester-project-yolov8\\spot.jpg'

yolov8_model_path = 'C:\\Users\\sakar\\Semester Project\\semester-project-yolov8\\runs\\pose\\train22\\weights\\last.pt'
sye_model_path = "C:/Users/sakar/Semester Project/simple-effective.pth"

def simple_yolo(yolov8_model_path, sye_model_path, image_path):
    outputstring2d = yolov8.predict(yolov8_model_path, image_path)

    outputstring3d = sye.run_prediction(outputstring2d, sye_model_path)

    print(outputstring3d)

simple_yolo(yolov8_model_path, sye_model_path, image_path)