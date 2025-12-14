# Invisibility Cloak (Gesture + Identity Controlled)

Real time invisibility cloak demo built with OpenCV and MediaPipe.  
The cloak effect is controlled by a custom trained gesture classifier, and it can be gated to only activate when my identity is detected.

## What it does
- Captures a clean background frame
- Segments people in the camera feed (MediaPipe Selfie Segmentation)
- Optionally detects my face and limits the cloak to only me (identity model)
- Recognizes hand gestures (gesture model) to control the cloak
- Replaces only the target person pixels with the saved background to create an invisibility effect

## Controls
Gesture controls (when identity allows it)
- peace: toggle cloak on or off
- open_palm: start background recapture timer (show again to cancel)
- thumbs_up: save a snapshot image

Keyboard fallback controls
- b: capture background immediately
- c: toggle cloak on or off
- q: quit

## Files
Main runtime
- cloak.py: runs the live demo, loads models, applies cloak effect, handles gestures and identity gating

Gesture pipeline
- collect_gesture_data.py: saves labeled hand landmark vectors to gesture_data.csv
- train_gesture_model.py: trains a Logistic Regression classifier and saves gesture_model.pkl
- gesture_data.csv: training data (label + 63 features)
- gesture_model.pkl: trained gesture classifier

Identity pipeline
- identity_features.py: converts FaceMesh landmarks into a normalized identity feature vector
- collect_identity_data.py: saves labeled identity vectors to identity_data.csv
- train_identity_model.py: trains a Logistic Regression classifier and saves identity_model.pkl
- identity_data.csv: training data (label + feature columns)
- identity_model.pkl: trained identity classifier

Other
- snapshots/: saved images when thumbs_up is triggered

## How the feature vectors work
Gesture features
- MediaPipe Hands returns 21 landmarks
- each landmark has x, y, z
- 21 x 3 = 63 features per gesture sample

Identity features
- MediaPipe FaceMesh returns many facial landmarks
- this project uses a small stable subset (eyes, nose, mouth corners, etc)
- those points are normalized (centered and scaled) then flattened into one vector

## Setup
Create a virtual environment and install dependencies.

Example
python -m venv .venv
source .venv/bin/activate

pip install opencv-python mediapipe numpy pandas scikit-learn joblib

## Training
1. Collect gesture data
python collect_gesture_data.py

2. Train gesture model
python train_gesture_model.py

3. Collect identity data
python collect_identity_data.py

4. Train identity model
python train_identity_model.py

If you do not train the identity model or remove identity_model.pkl, the project will run without identity gating.

## Run the cloak demo
python cloak.py

Recommended flow
- press b to capture a clean background while you step out of frame
- step back in
- use gestures to control the cloak

## Notes and limitations
- Lighting, fast motion, and poor landmark tracking can reduce accuracy
- If multiple hands are detected at once, control can be disabled to avoid ambiguity
- If identity gating is enabled, gestures only trigger when my face is detected

## Tech used
- OpenCV for webcam capture, drawing, and compositing
- MediaPipe Hands for hand landmarks
- MediaPipe FaceMesh for face landmarks
- MediaPipe Selfie Segmentation for person segmentation mask
- scikit-learn Logistic Regression for gesture classification and identity classification
- joblib for saving and loading trained models as pkl files
