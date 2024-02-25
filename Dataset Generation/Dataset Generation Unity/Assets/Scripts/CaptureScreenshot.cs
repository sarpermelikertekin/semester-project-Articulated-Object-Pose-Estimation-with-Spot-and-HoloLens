using UnityEngine;

public class CaptureScreenshot : MonoBehaviour
{
    public static CaptureScreenshot Instance { get; private set; }

    public bool capture;
    public string fileName;
    public int pictureIndex;
    public PictureFormat pictureFormat;

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

    public void TakeScreenshot(string path)
    {
        if (capture)
        {
            string formatExtension = GetFormatExtension(pictureFormat);
            fileName = $"{pictureIndex}.{formatExtension}";
            ScreenCapture.CaptureScreenshot(path + fileName);
        }
    }

    private string GetFormatExtension(PictureFormat format)
    {
        switch (format)
        {
            case PictureFormat.PNG:
                return "png";
            case PictureFormat.JPEG:
                return "jpg";
            case PictureFormat.EXR:
                return "exr";
            default:
                return "png";
        }
    }
}

public enum PictureFormat
{
    PNG,
    JPEG,
    EXR
}

