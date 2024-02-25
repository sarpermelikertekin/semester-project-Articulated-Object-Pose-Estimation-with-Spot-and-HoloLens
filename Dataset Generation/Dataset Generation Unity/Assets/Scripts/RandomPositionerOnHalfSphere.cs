using UnityEngine;

public class RandomPositionerOnHalfSphere : MonoBehaviour
{
    public GameObject objectToObserve;

    [Header("Sphere")]
    public float minSphereRadius;
    public float maxSphereRadius;

    [Header("Tilt")]
    public bool isTilted;
    public float cameraTiltValue;
    public bool randomizeXTilt;
    public bool randomizeYTilt;
    public bool randomizeZTilt;

    [Header("Clamp Camera Height")]
    public float maxY;
    public float minY;

    private void Start()
    {

    }

    public void RepositionObject()
    {
        // Generate a random radius between minSphereRadius and maxSphereRadius
        float randomRadius = Random.Range(minSphereRadius, maxSphereRadius);

        // Generate a random position on a half-sphere
        Vector3 newPosition = RandomPositionOnHalfSphere(randomRadius);

        //Clamp the camera height
        newPosition = new Vector3(newPosition.x, Mathf.Clamp(newPosition.y, minY, maxY), newPosition.z);
        transform.position = newPosition;

        // Always look at the object
        transform.LookAt(objectToObserve.transform);

        // Randomly set isTilted to true or false
        isTilted = Random.value > 0.5f;

        //Conditionally tilt the camera's rotation
        if (isTilted)
        {
            float xRotation = randomizeXTilt ? Random.Range(-cameraTiltValue, cameraTiltValue) : cameraTiltValue;
            float yRotation = randomizeYTilt ? Random.Range(-cameraTiltValue, cameraTiltValue) : cameraTiltValue;
            float zRotation = randomizeZTilt ? Random.Range(-cameraTiltValue, cameraTiltValue) : cameraTiltValue;

            // Apply the randomized rotations
            transform.Rotate(xRotation, yRotation, zRotation);
        }
    }

    private Vector3 RandomPositionOnHalfSphere(float radius)
    {
        // Random angle in radians for half-sphere (upper hemisphere)
        float theta = Random.Range(0, Mathf.PI); // Longitude
        float phi = Random.Range(0, Mathf.PI / 2); // Latitude limited to upper hemisphere

        // Convert spherical coordinates to Cartesian coordinates
        float x = radius * Mathf.Sin(phi) * Mathf.Cos(theta);
        float y = radius * Mathf.Sin(phi) * Mathf.Sin(theta);
        float z = radius * Mathf.Cos(phi);

        return new Vector3(x, y, z);
    }
}
