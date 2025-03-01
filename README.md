# Fire Detection Video Processing

This project utilizes a YOLO model to detect fire in video files, annotating each frame where fire is detected and producing an output video with these annotations.

## Requirements

- Python 3.8 or higher
- `torch`
- `torchvision`
- `ultralytics`
- `opencv-python`
- `numpy`
- `gradio`

Install the required packages using:

```bash
pip install torch torchvision ultralytics opencv-python numpy gradio
```

## Usage

1. **Model Placement**: Ensure the YOLO model file (`last.pt`) is located in the same directory as the Python script.

2. **Running the Script**: Execute the Python script to start the Gradio web interface:

   ```bash
   python app.py
   ```

3. **Using the Web Interface**: Once the script is running, a web interface will launch. Upload a video file through this interface. The system will process the video, detect any instances of fire, and provide an annotated video as output.

## Notes

- **Device Selection**: The script automatically selects the best available device for processing—GPU (`cuda`), Apple's Metal Performance Shaders (`mps`), or CPU (`cpu`).

- **Dependencies**: Ensure all required packages are installed. If you encounter issues, verify that your environment meets the specified requirements.

- **Model Compatibility**: The `last.pt` model file should be compatible with the version of the `ultralytics` package used. If you experience errors, consider updating or downgrading the package or model file accordingly.

For further information or troubleshooting, refer to the documentation of the respective packages or seek assistance from the community.
