using UnityEngine;

// This Script Should be attached to the joints of the objects
// 
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
            Gizmos.color = gizmoColor;

            Gizmos.DrawSphere(transform.position, gizmoSize);

            // Optional: Draw the name and transform info near the gizmo
#if UNITY_EDITOR
            string label = gizmoName + "\n" + transform;
            UnityEditor.Handles.Label(transform.position, label);
#endif
        }
    }
}

