using UnityEngine;
using System.Collections.Generic;

public class KeypointSpheresManager : MonoBehaviour
{
    public static KeypointSpheresManager Instance { get; private set; }
    public GameObject spherePrefab;
    public Material lineMaterial;

    private List<GameObject> keypoints = new List<GameObject>();
    private List<int[]> linePairs = new List<int[]> {
        new int[] {0, 1}, new int[] {1, 4}, new int[] {1, 7}, new int[] {7, 10}, new int[] {4, 10},
        new int[] {1, 2}, new int[] {2, 3}, new int[] {4, 5}, new int[] {5, 6},
        new int[] {7, 8}, new int[] {8, 9}, new int[] {10, 11}, new int[] {11, 12}
    };

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject); // Optional
        }
        else if (Instance != this)
        {
            Destroy(gameObject);
        }
    }

    public void CreateKeypointsFromString(string keypointsString)
    {
        // Clear previous keypoints and lines
        foreach (var keypoint in keypoints)
        {
            Destroy(keypoint);
        }
        keypoints.Clear();

        var positions = ParseStringToVector3List(keypointsString);
        foreach (var position in positions)
        {
            GameObject sphere = Instantiate(
                spherePrefab ? spherePrefab : GameObject.CreatePrimitive(PrimitiveType.Sphere),
                position,
                Quaternion.identity
            );
            sphere.transform.localScale = new Vector3(0.2f, 0.2f, 0.2f); // Setting scale for 10cm radius
            keypoints.Add(sphere);
        }

        DrawLinesBetweenKeypoints();
    }

    private void DrawLinesBetweenKeypoints()
    {
        foreach (var pair in linePairs)
        {
            if (pair.Length == 2 && pair[0] < keypoints.Count && pair[1] < keypoints.Count)
            {
                DrawLine(keypoints[pair[0]].transform.position, keypoints[pair[1]].transform.position);
            }
        }
    }

    private void DrawLine(Vector3 start, Vector3 end)
    {
        GameObject lineObj = new GameObject("Line");
        lineObj.transform.SetParent(transform);
        LineRenderer lineRenderer = lineObj.AddComponent<LineRenderer>();
        lineRenderer.material = lineMaterial;
        lineRenderer.startWidth = 0.01f;
        lineRenderer.endWidth = 0.01f;
        lineRenderer.SetPosition(0, start);
        lineRenderer.SetPosition(1, end);
    }

    private List<Vector3> ParseStringToVector3List(string str)
    {
        List<Vector3> positions = new List<Vector3>();
        // Removing the outer brackets
        str = str.Trim('[', ']');
        // Splitting into individual position strings
        foreach (var posStr in str.Split(new string[] { "], [" }, System.StringSplitOptions.RemoveEmptyEntries))
        {
            var coords = posStr.Split(',');
            if (coords.Length == 3)
            {
                if (float.TryParse(coords[0], out float x) && float.TryParse(coords[1], out float y) && float.TryParse(coords[2], out float z))
                {
                    positions.Add(new Vector3(x, y, z));
                }
            }
        }
        return positions;
    }
}
