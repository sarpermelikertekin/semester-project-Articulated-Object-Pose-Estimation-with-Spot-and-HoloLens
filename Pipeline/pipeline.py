import simple_yet_effective as sye
import yolov8

def simple_yolo(yolov8_model_path, sye_model_path, image_path):
    outputstring2d = yolov8.predict(yolov8_model_path, image_path)

    outputstring3d = sye.run_prediction(outputstring2d, sye_model_path)

    print(outputstring3d)

    return outputstring3d

## More models can be added as functions to this script to select from the available ones
## The script which are present here can be used in server script to process images