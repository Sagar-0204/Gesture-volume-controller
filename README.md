# Gesture-Controlled Volume Controller

This project is an **AI-powered system volume controller** that uses your **hand gestures** in real-time via webcam. Built with OpenCV, MediaPipe, and Pycaw, it adjusts the system volume, mutes/unmutes, and locks/unlocks control using intuitive hand movements.

---

## Features

-  Control volume by pinching your **thumb and index finger**
-  **Mute/unmute** with **index & middle finger touch**
-  **Lock/unlock** volume control with **thumb & pinky finger gesture**
-  Real-time hand tracking with **MediaPipe Hands**
-  Webcam-based UI using **OpenCV**
-  System audio interface via **pycaw**

---

## Files

- `volume.py` – Main Python script that runs the gesture recognition and controls volume.

---

## Requirements

Install dependencies using:

```bash
pip install opencv-python mediapipe pycaw comtypes numpy
