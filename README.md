# Semester Project: Articulated Object Pose Estimation with Spot and HoloLens
Semester Project with CVG Lab where we introduce a novel method for 3d pose estimation of quadruped robots using latest YOLOv8, a "Simple Yet Effective" 2D-3D Mapping Network, a custom Transformer for pose detection (Poseformer), dataset generation pipeline to finetune the models in Unity Framework as well as data validators, MR UI for Hololens using MRTK2 and a communication pipeline to send and process the images as well as receive the processed answer to be displayed in MR environment.

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/90496aff-5b4f-426d-a3be-7ec04833e7ec)

## Introduction and Repository Structure
For each network we created a playground jupyter notebooks file to train and quickly test the models. To use them in the pipeline with HoloLens we have a directory called pipeline and in it we have pipeline.py to use the models in inference time.

Furthermore for MR UI and Dataset Generation we used Unity Framework and in MR UI we used MRTK2 for creating the UI for HoloLens.

## Installation and Required Libraries
We used Unity version 2021.3.33f1 in this project but any other stable version would be suitable as well. One can use LTS version to make sure it will work properly. For MR UI please check it with MRTK2 to see if it is compatible. If not you might encounter build issues.

In Python we created a new environment and installed following libraries:
- PyTorch
- Numpy
- CV2
- PIL
- matplotlib
- ultralytics

For the traiend Simple Yet Effective Network, Poseformer and the Datasets we have generated, please contact to us.

## Dataset Generation Pipeline
To use and finetune our models we introduced a novel pipeline for dataset generation in Unity Framework. In our pipeline we take a screenshot of the current game view with the quadruped robot in it, then create the corresponding annotations for COCO Dataset format to be fed into YOLOv8 and 2D as well 3D positions of the Keypoints saved in a JSON file. We use a simple plane and a robot model. In each iteration we rotate the camera around the Spot, apply a random texture on plane and skybox, take a screenshot and generate the corresponding annotations and JSON files.

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/78e0a860-15e4-40a1-8caa-2bec0dc9ee32)
![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/51cde459-a34b-4813-8b33-6eb680bee5dd)

Furthermore to handle visibility flag, one needs to put colliders in the object as well as place Keypoint Objects with Keypoint Renderer Script attached to them to detect the visibility. We send a ray from camera to each Joint and check if it collides with the corresponding keypoint or not. Also for the class distinction, one has to tag the object accordingly. Currently we have 2 classes "Spot" and "AnyMAL" and 13 keypoints with 1 for head and 3 keypoints for each leg.

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/11306267-62b9-499c-85ed-1bb6569af262)

Furthermore we have 6 poses for Spot and one for AnyMAL. They both have idle and spot has Off, Gait 1 - 2, Flex 1 - 2. Here is an overview of the poses:

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/ccee7b4d-3a31-4cd5-9af6-18ab123b250e)

We also created a scene for video capture using Cinemachine.

Finally here is an overview of assets we used for this project
- [Sketchfab For Unity to import the Sketchfab models into Unity (Spot and AnyMAL models)](https://assetstore.unity.com/packages/tools/input-management/sketchfab-for-unity-1430)
- [YughuesFreeMetalMaterials for plane textures](https://assetstore.unity.com/packages/2d/textures-materials/metals/yughues-free-metal-materials-12949)
- [Ground textures pack for plane textures](https://assetstore.unity.com/packages/2d/textures-materials/floors/yughues-free-ground-materials-13001#content)
- [AllSkyFree for skybox textures](https://assetstore.unity.com/packages/2d/textures-materials/sky/allsky-free-10-sky-skybox-set-146014)

## Dataset Generation Pipeline - Data Analysis Tools
Furthermore we created a helper jupyter notebook file to validate the datasets we created. It has 3 functionalities.

### Bounding Box and Keypoint Validator
It is used to validate the annotations. It simply places the points from annotations to the image

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/56c64aac-2f6d-433a-9184-fdc99ab30bfe)

### Bounding Box and Keypoint "Mask"
It is used to validate the data distributions. We simply visualize the bounding box and keypoints on a black canvas. The idea is to have an idea of where the object is in the images as well as the single keypoints. It is used to give us an intuition of the data we generated and if we need more variations, when it comes to position of the object.

![654_bbox](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/73138f03-295b-43b0-b04e-4347b5f0bb0a)
![654_keypoints](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/41b8bbbe-d134-409e-9cff-11d2736b76fa)

### Bounding Box and Keypoints Histogram
It is used to validate the data distributions again. Here the idea is to have an overview of the whole dataset. We plot the positions of each bounding box center and every keypoint. As well as generate heatmaps and histograms for the size of bounding box, center of bounding box and keypoint locations

![distribution_of_bb_centers](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/19095625-fb2b-4c9e-8102-c64b2d64adbf)
![distribution_of_keypoint_1](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/74cbf26f-2bd1-4f33-ab2e-7aca49ed235d)
![heatmap_of_bb_centers](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/39e13692-50a1-457c-a958-a92e4852fbb1)
![heatmap_of_keypoint_1](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/ea6bd57f-9f5f-4336-b2fa-ebb02b025f8f)
![histogram_of_bb_areas](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/b8a107ac-0585-4b48-a840-1d1fc46dd4bf)

## Communication Pipeline and MR UI
To perform the ML computations we used an off-device approach where we capture the image from HoloLens, send it to the server and process it with our Network. After that we receive the 3D keypoint locations as a string and we display them in MR space.

Here is how it looks like in MR Space

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/eb6aac49-9c3a-42ef-83d6-cf032c107d14)

Furthermore we introduced a shell script to enable extension of this project. New models can be easily added to pipeline folder (in inference) and one can easily build a dunction in pipeline script to predict and use in server script.

## YOLOv8
With the latest YOLOv8 it is possible to perform pose estimation task. To finetune the YOLOv8 we used synthetic data we generated from the Unity Dataset Generator and to close the domain gap we manually annotated 50 real images and included them into the dataset at the end.

YOLOv8 is accepting the COCO Dataset annotation. In our project we used the following format: 

[class_id bounding_box_center_x bounding_box_center_y bounding_box_width bounding_box_height keypoint_n_x keypoint_n_y keypoint_n_visibility_flag] with visibility flag being 1: not visible and 2: visible

We use YOLOv8 to predict 2D keypoint locations. Here are some examples of predictions we got from YOLOv8.

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/1b413b26-f4f8-420d-b227-e33fe83ad1fd)
![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/87884a8a-ca38-45ad-902c-b339a6b91690)

- [We got inspired from this tutorial video](https://www.youtube.com/watch?v=gA5N54IO1ko&t=548s)
- [More About Pose Estimation with YOLOv8](https://docs.ultralytics.com/tasks/pose/)

## 2D-3D Mapping: Simple Yet Effective Network
For 2D-3D Mapping, we used the network proposed in the paper ["A simple yet effective baseline for 3d human pose estimation"](https://arxiv.org/pdf/1705.03098v2.pdf). It is a network used to extract 3D keypoints given 2D keypoint extractions, which basically maps the 2D detections into 3D. We used PyTorch to implement the network.

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/3b5e9430-2123-42f4-99e1-acb732a81537)

We use Simple Yet Effective Network, to map the 2D keypoint predictions from YOLOv8 

![Networks back to back](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/345050d5-40ca-4f4d-9677-2d445f118ed3)

An example of predictions we got from Simple Yet Effective Network

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/99de42d7-497c-4c85-9ce5-bc27a7c806d5)

## Custom Transformer for pose detection (Poseformer)
Furthermore we introduced a unique transformer which should have been capable of performing 2D-3D Mapping task. Since both 2D and 3D positions can be treated as a sequence, we treated this as a seq2seq task and therefore used a Transformer. In this Transformer we take the 2D keypoint and the image itself as input and as output we return corresponding 3D mapping. We used PyTorch to implement the network.

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/3695c1d3-f510-402f-9563-2f486756cfd4)

An example of predictions we got from Poseformer. It performs significantly worse than the Simple Yet Effective Network, which is quite interesting outcome. This result highlights again how powerful the Simple Yet Effective Network is.

![image](https://github.com/sarpermelikertekin/semester-project-Articulated-Object-Pose-Estimation-with-Spot-and-HoloLens/assets/49168444/8f4dd1e6-64b0-495f-9366-c0487f255f46)
