# revoltact-wlf
> Video Face Detection with Super-Resolution Enhancement

## Overview
This script is designed to run in a Google Colab environment and performs face detection on video frames using a combination of super-resolution and the DeepFace library. The code first enhances the resolution of each video frame using a pre-trained ESRGAN model from TensorFlow Hub, and then it applies face detection to the upscaled frames. Detected faces are displayed at the end of the process.

## Dependencies
- deepface: A Deep Learning face recognition and demography library for Python.
- tqdm: A library for displaying progress bars in Python loops.
- tensorflow-hub: A library for loading and using TensorFlow models from TensorFlow Hub.
- OpenCV (cv2): An open-source computer vision and machine learning software library.
- matplotlib: A plotting library for Python and its numerical mathematics extension, NumPy.
- NumPy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## Installation
The script installs the required libraries using `pip`. TensorFlow Hub is used to load the ESRGAN model.

## Workflow
1. The user is prompted to upload a video file to the Colab environment.
2. The video is loaded, and its frames are processed one by one.
3. Each frame is upscaled using the ESRGAN model, enhancing its resolution.
4. The upscaled frame is converted from BGR to RGB color space.
5. The DeepFace library's `extract_faces` function is used to detect faces in the RGB frame.
6. Detected faces are collected and counted.
7. A progress bar is displayed to show the progress of the frame processing.
8. After all frames are processed, the total number of frames and detected faces are printed.
9. Finally, the detected faces are displayed using `matplotlib`.

## Functions
- `preprocess_frame(frame)`: Prepares a video frame for super-resolution by converting it to RGB, normalizing pixel values, and adding a batch dimension.
- `upscale_frame(frame, model)`: Applies the ESRGAN model to upscale a preprocessed frame, then converts it back to BGR color space.
- `extract_and_display_faces(video_path)`: The main function that orchestrates the loading of the video, frame processing, face detection, and display of results.

## Usage
To use this script, simply run it in a Google Colab notebook with a GPU runtime for better performance. The script will handle the rest, from prompting for video upload to displaying the detected faces.

## Notes
- Ensure that the Colab runtime is set to use a GPU for faster processing. This can be done by going to `Runtime` > `Change runtime type` and selecting `GPU` as the hardware accelerator.
- The ESRGAN model is used to improve the quality of video frames, which can help in detecting faces more accurately, especially in low-resolution videos.
- The `extract_faces` function from DeepFace is used instead of the deprecated `detectFace` function, following the library's updated usage recommendations.
