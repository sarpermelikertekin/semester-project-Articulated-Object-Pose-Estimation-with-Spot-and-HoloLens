using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;

public class SpotDatasetGenerator : MonoBehaviour
{
    public static SpotDatasetGenerator Instance { get; private set; }

    CaptureScreenshot captureScreenshot;

    string baseLocation = @"C:\Users\sakar\Semester Project\Spot Datasets\";

    [Header("File Location and Control")]
    public string datasetFileName;
    public bool splitSet;
    public KeyCode startingKey;

    [Header("Randomization Controls")]
    public bool randomizeTextures = true;
    public bool randomizeSkybox = true;
    public bool randomizeLighting = true;

    [Header("Normalization Options")]
    public bool normalize;
    public int xNormalization;
    public int yNormalization;

    [Header("Image Capture Settings")]
    public int numberOfImages;
    public float dataPointWaitTime;
    public int counter;

    private int totalImagesCount;
    private int trainImagesCount;
    private int testImagesCount = 10;

    [Header("Keypoints")]
    public GameObject[] keypoints;

    [Header("Camera")]
    public GameObject cameraObject;

    [Header("Robot Poses")]
    public GameObject robotPoses;

    [Header("Directional Light")]
    public GameObject directionalLight;
    public int directionalLightRotationXMin;
    public int directionalLightRotationXMax;
    public int directionalLightRotationYMin;
    public int directionalLightRotationYMax;

    [Header("Textures")]
    public Material[] materials;
    public GameObject plane;
    public Material[] skyboxes;

    [System.Serializable]
    public class KeypointPose
    {
        public string name;
        public Vector3 position;
        public Vector3 rotation;
    }

    [System.Serializable]
    public class KeypointData
    {
        public string name;
        public Vector2 position;
        public int isVisible;
    }

    [System.Serializable]
    public class Dataset
    {
        public int Class = 0;
        public Vector2 boundingBoxCenter;
        public Vector2 boundingBoxSize;
        public List<KeypointData> keypoints;
    }

    [System.Serializable]
    public class KeypointPoseList
    {
        public List<KeypointPose> poses;
    }

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else if (Instance != this)
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        captureScreenshot = CaptureScreenshot.Instance;
        baseLocation += datasetFileName + "\\";

        totalImagesCount = numberOfImages;
        trainImagesCount = (int)(totalImagesCount * 0.8);
    }

    void Update()
    {
        if (Input.GetKeyDown(startingKey))
        {
            StartCoroutine(StartGeneratingDataSet(dataPointWaitTime));
        }
    }

    // Checks if a keypoint is visible from the camera
    private bool IsKeypointVisible(GameObject keypoint)
    {
        Vector3 directionToKeypoint = keypoint.transform.position - cameraObject.GetComponent<Camera>().transform.position;

        RaycastHit hit;
        if (Physics.Raycast(cameraObject.GetComponent<Camera>().transform.position, directionToKeypoint, out hit))
        {
            return hit.transform == keypoint.transform;
        }
        return false;
    }

    private float MirrorY(float originalY)
    {
        float screenHeight = Screen.height;
        return screenHeight - originalY;
    }

    public IEnumerator StartGeneratingDataSet(float seconds)
    {
        yield return new WaitForSeconds(seconds);

        // Random texture
        if (randomizeTextures)
        {
            plane.GetComponent<MeshRenderer>().material = materials[Random.Range(0, materials.Length)];
        }


        // Random Rotation for the lighting
        if (randomizeLighting)
        {
            directionalLight.transform.eulerAngles = new Vector3(
                Random.Range(directionalLightRotationXMin, directionalLightRotationXMax),
                Random.Range(directionalLightRotationYMin, directionalLightRotationYMax),
                0f);
        }

        // Random skybox
        if (randomizeSkybox)
        {
            RenderSettings.skybox = skyboxes[Random.Range(0, skyboxes.Length)];
        }

        // Deactivate all children of robotPoses
        foreach (Transform child in robotPoses.transform)
        {
            child.gameObject.SetActive(false);
        }

        // Activate one child of robotPoses randomly
        int childrenCount = robotPoses.transform.childCount;
        GameObject activeRobotPose = null;
        if (childrenCount > 0)
        {
            int randomIndex = UnityEngine.Random.Range(0, childrenCount);
            activeRobotPose = robotPoses.transform.GetChild(randomIndex).gameObject;
            activeRobotPose.SetActive(true);
        }

        // Find active keypoints with the KeypointRenderer script
        List<GameObject> activeKeypoints = new List<GameObject>();
        KeypointRenderer[] componentsWithScript = FindObjectsOfType<KeypointRenderer>(true);
        foreach (var component in componentsWithScript)
        {
            if (component.gameObject.activeInHierarchy)
            {
                activeKeypoints.Add(component.gameObject);
            }
        }
        keypoints = activeKeypoints.ToArray();

        keypoints = activeKeypoints.OrderBy(go => go.GetComponent<KeypointRenderer>().priority).ToArray();

        cameraObject.GetComponent<Camera>().GetComponent<RandomPositionerOnHalfSphere>().RepositionObject();

        float buffer = 25f;
        float minX = float.MaxValue;
        float maxX = float.MinValue;
        float minY = float.MaxValue;
        float maxY = float.MinValue;

        if (normalize)
        {
            xNormalization = Screen.width;
            yNormalization = Screen.height;
        }
        
        foreach (GameObject keypoint in keypoints)
        {
            Vector3 screenPosition = cameraObject.GetComponent<Camera>().WorldToScreenPoint(keypoint.transform.position);

            // Flip the y-coordinate around the center of the screen
            screenPosition = new Vector3(screenPosition.x, MirrorY(screenPosition.y));

            bool isVisible = IsKeypointVisible(keypoint);

            minX = Mathf.Min(minX, screenPosition.x);
            maxX = Mathf.Max(maxX, screenPosition.x);
            minY = Mathf.Min(minY, screenPosition.y);
            maxY = Mathf.Max(maxY, screenPosition.y);
        }

        // Add buffer to the bounding box
        minX -= buffer;
        maxX += buffer;
        minY -= buffer;
        maxY += buffer;

        minX /= xNormalization;
        maxX /= xNormalization;
        minY /= yNormalization;
        maxY /= yNormalization;

        Vector2 boundingBoxCenter = new Vector2((maxX + minX) / 2, (maxY + minY) / 2);
        Vector2 boundingBoxSize = new Vector2(maxX - minX, maxY - minY);

        // Check tag of the active robot pose and set class accordingly
        int classForDataset = 0;
        if (activeRobotPose != null)
        {
            if (activeRobotPose.tag == "Spot")
                classForDataset = 0;
            else if (activeRobotPose.tag == "AnyMAL")
                classForDataset = 1;
        }

        string baseSetPath = "";

        if (splitSet)
        {
            string setCategory;
            if (captureScreenshot.pictureIndex < trainImagesCount)
                setCategory = "train";
            else if (captureScreenshot.pictureIndex < totalImagesCount)
                setCategory = "val";
            else
                setCategory = "test";

            baseSetPath = Path.Combine(baseLocation, setCategory);
        }
        else
        {
            baseSetPath = Path.Combine(baseLocation, "");
        }

        string screenshotFileName = Path.Combine(baseSetPath, "images\\");
        string labelFileName = Path.Combine(baseSetPath, "labels\\");
        string poseDirectory = Path.Combine(baseSetPath, "mapping_3d\\");
        string datasetDirectory = Path.Combine(baseSetPath, "mapping_2d\\");

        EnsureDirectoryExists(screenshotFileName);
        EnsureDirectoryExists(labelFileName);
        EnsureDirectoryExists(poseDirectory);
        EnsureDirectoryExists(datasetDirectory);

        captureScreenshot.TakeScreenshot(screenshotFileName);

        // Data string creation
        string dataString = classForDataset + " " + boundingBoxCenter.x + " " + boundingBoxCenter.y + " " + boundingBoxSize.x + " " + boundingBoxSize.y;
        foreach (GameObject keypoint in keypoints)
        {
            Vector3 screenPosition = cameraObject.GetComponent<Camera>().WorldToScreenPoint(keypoint.transform.position);
            screenPosition = new Vector3(screenPosition.x, MirrorY(screenPosition.y));
            bool isVisible = IsKeypointVisible(keypoint);
            int visibilityFlag = isVisible ? 2 : 1;
            dataString += " " + screenPosition.x / xNormalization + " " + screenPosition.y / yNormalization + " " + visibilityFlag;
        }

        if (!Directory.Exists(labelFileName))
        {
            Directory.CreateDirectory(labelFileName);
        }

        // Write data to text file
        string txtFileName = Path.ChangeExtension(labelFileName + captureScreenshot.pictureIndex, ".txt");
        File.WriteAllText(txtFileName, dataString);

        KeypointPoseList keypointPoseList = new KeypointPoseList { poses = new List<KeypointPose>() };
        Dataset dataset = new Dataset()
        {
            Class = classForDataset,
            boundingBoxCenter = boundingBoxCenter,
            boundingBoxSize = boundingBoxSize,
            keypoints = new List<KeypointData>()
        };

        foreach (GameObject keypoint in keypoints)
        {
            Vector3 relativePosition = cameraObject.GetComponent<Camera>().transform.InverseTransformPoint(keypoint.transform.position);

            // Add 6D pose data
            keypointPoseList.poses.Add(new KeypointPose
            {
                name = keypoint.name,
                position = relativePosition,
                rotation = keypoint.transform.eulerAngles
            });

            // Add 2D data with visibility
            Vector3 screenPosition = cameraObject.GetComponent<Camera>().WorldToScreenPoint(keypoint.transform.position);
            screenPosition = new Vector3(screenPosition.x, MirrorY(screenPosition.y));
            dataset.keypoints.Add(new KeypointData
            {
                name = keypoint.name,
                position = new Vector2(screenPosition.x / xNormalization, screenPosition.y / yNormalization),
                isVisible = IsKeypointVisible(keypoint) ? 2 : 1
            });
        }

        // Serialize to JSON
        string poseJson = JsonUtility.ToJson(keypointPoseList, true);
        string datasetJson = JsonUtility.ToJson(dataset, true);

        string poseFilePath = poseDirectory + captureScreenshot.pictureIndex + ".json";
        string datasetFilePath = datasetDirectory + captureScreenshot.pictureIndex + ".json";
        File.WriteAllText(poseFilePath, poseJson);
        File.WriteAllText(datasetFilePath, datasetJson);

        ++captureScreenshot.pictureIndex;

        if (captureScreenshot.pictureIndex < totalImagesCount + testImagesCount)
        {
            StartCoroutine(StartGeneratingDataSet(seconds));
        }
    }

    private void EnsureDirectoryExists(string path)
    {
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
    }
}
