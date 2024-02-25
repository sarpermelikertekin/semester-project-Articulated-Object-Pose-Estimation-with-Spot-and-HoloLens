using UnityEngine;

public class KeypointRenderer : MonoBehaviour
{
    public string gizmoName;
    public Color gizmoColor;
    public float gizmoSize;

    public bool drawGizmos;

    public int priority;

    private void Awake()
    {
        gizmoName = gameObject.name;
    }

    private void OnDrawGizmos()
    {
        if (drawGizmos)
        {
            // Set the color of the gizmo
            Gizmos.color = gizmoColor;

            // Draw a sphere at the position of the GameObject
            Gizmos.DrawSphere(transform.position, gizmoSize);

            // Optional: Draw the name and transform info near the gizmo
#if UNITY_EDITOR
            string label = gizmoName + "\n" + transform;
            UnityEditor.Handles.Label(transform.position, label);
#endif
        }
    }
}

