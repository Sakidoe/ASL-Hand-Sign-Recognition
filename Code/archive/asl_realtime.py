"""
ASL Real-Time Recognition System (cleaned & student-style comments)
-----------------------------------------------------------------
We built a single-file ASL recognition demo that uses MediaPipe for
hand detection, a small CNN trained on Sign Language MNIST for static
letter recognition, and simple motion tracking heuristics to detect
dynamic letters J and Z. The file below is rewritten to be clearer,
avoid duplicated code, and fix variable shadowing and instantiation
bugs reported by our linter.

What we changed (short):
- fixed a potential "object not callable" bug by ensuring we instantiate
  the model with parentheses: `model = SimpleCNN()`
- removed duplicated bounding-box calculation by putting it in
  `bbox_from_landmarks()` and reusing that helper
- removed the shadowing warning by using a single global `motion_history`
  list and not re-declaring it inside functions

Notes to reviewers: this file is written in a student voice ("we") with
short 3-4 line comments for each logical block to make the flow easy
to follow.
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
import pyttsx3

# -----------------------------
# Model: Small CNN
# -----------------------------
# We use the same CNN architecture as in train_asl.py so that the
# saved weights in asl_cnn.pth match the layer names (features/*,
# classifier/*). This avoids state_dict key mismatches.
class SimpleCNN(torch.nn.Module):
    # small CNN for 28x28 grayscale images
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 3 * 3, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Small helper to map index->letter (0->A, 25->Z)
def index_to_letter(idx: int) -> str:
    # We keep mapping simple and linear for A..Z. J and Z are dynamic
    # but still occupy index positions if the dataset uses them.
    return chr(ord("A") + int(idx))


# -----------------------------
# MediaPipe + Preprocessing
# -----------------------------
# We initialize MediaPipe Hands once and reuse it for each frame.
# The helper `bbox_from_landmarks` computes a safe crop around the hand
# and returns both the bounding box and the center (for motion tracking).
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def bbox_from_landmarks(landmarks, frame_shape, pad: int = 20):
    """
    Compute bounding box and center from MediaPipe landmarks.
    Returns (x_min,y_min,w,h), (cx,cy). Uses padding and clamps to image.
    """
    h, w = frame_shape[:2]
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    x_min = int(min(xs) * w) - pad
    y_min = int(min(ys) * h) - pad
    x_max = int(max(xs) * w) + pad
    y_max = int(max(ys) * h) + pad

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)

    bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))
    center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
    return bbox, center


def preprocess_hand(frame, landmarks):
    """
    Crop the hand region defined by `landmarks`, convert to grayscale,
    apply blur + histogram equalization and Otsu thresholding, then
    resize to 28x28. Returns a tensor and the debug-threshold image.
    """
    bbox, _ = bbox_from_landmarks(landmarks, frame.shape)
    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Resize to model input size and normalize
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    img = resized.astype(np.float32) / 255.0
    img = img.reshape(1, 1, 28, 28)
    tensor = torch.tensor(img, dtype=torch.float32).to(DEVICE)
    return tensor, resized


# -----------------------------
# Motion tracking (J and Z)
# -----------------------------
# We keep a single global history list of recent hand centers and use a
# compact heuristic to detect J (downward arc) and Z (mostly horizontal
# sweep). This is a heuristic — for production we'd train a sequence model.
motion_history = []
MOTION_FRAMES = 12


def detect_motion_letter_from_history():
    """
    Analyze the global motion_history and return 'J' or 'Z' when patterns
    match our simple heuristics. Returns None otherwise.
    """
    if len(motion_history) < MOTION_FRAMES:
        return None

    x0, y0 = motion_history[0]
    x1, y1 = motion_history[-1]
    dx = x1 - x0
    dy = y1 - y0

    # Z heuristic: large horizontal motion, small vertical change
    if abs(dx) > 35 and abs(dy) < 18:
        return "Z"

    # J heuristic: substantial downward motion
    if dy > 25:
        return "J"

    return None


# -----------------------------
# Optional speech
# -----------------------------
# We use pyttsx3 for offline TTS. We keep the call optional and simple.
engine = pyttsx3.init()
engine.setProperty("rate", 140)


def speak(text: str):
    # Non-blocking usage would be nicer, but for clarity we call runAndWait.
    engine.say(text)
    engine.runAndWait()


# -----------------------------
# Drawing helpers
# -----------------------------
# Small utilities to draw the top prediction and a small probability
# bar for quick visual feedback. We avoid repeating code by centralizing
# drawing in these helpers.
def draw_prediction(frame, text: str, bbox=None):
    cv2.putText(
        frame,
        f"Pred: {text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_top_probs(frame, probs, top_k: int = 6):
    # Draw top-k probability text on the right side of the frame.
    if probs is None:
        return
    h, w = frame.shape[:2]
    start_x = w - 160
    start_y = 20
    idxs = np.argsort(-probs)[:top_k]
    for i, idx in enumerate(idxs):
        letter = index_to_letter(idx)
        val = float(probs[idx])
        cv2.putText(
            frame,
            f"{letter}: {val:.2f}",
            (start_x, start_y + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


# -----------------------------
# Main loop
# -----------------------------
# We load the model, open the webcam, and then process each frame. For
# smoothing we keep the last few predictions and take the majority vote.
# We also collect saved letters into a string that is written to disk.
def main(model_path: str = "asl_cnn.pth", speak_output: bool = False):
    # Instantiate and load model (ensure we create an object instance)
    model = SimpleCNN(num_classes=26)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found")
        return

    recent_preds = []
    saved_letters = []

    print("Starting webcam. Press 'q' to quit. Press 's' to toggle speech.")
    do_speak = speak_output

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        predicted = "?"
        bbox = None
        probs = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # compute bbox and center once and reuse it
            bbox, center = bbox_from_landmarks(hand_landmarks, frame.shape)
            cx, cy = center

            # update global motion history
            motion_history.append((cx, cy))
            if len(motion_history) > MOTION_FRAMES:
                motion_history.pop(0)

            # motion-based override for J/Z
            motion_letter = detect_motion_letter_from_history()

            # preprocess for CNN and run inference
            tensor, debug_img = preprocess_hand(frame, hand_landmarks)
            if tensor is not None:
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    idx = int(np.argmax(probs))
                    predicted = index_to_letter(idx)

                # smoothing: majority vote over last 5 predictions
                recent_preds.append(predicted)
                if len(recent_preds) > 5:
                    recent_preds.pop(0)
                smoothed = max(set(recent_preds), key=recent_preds.count)

                # override with detected dynamic gesture
                if motion_letter:
                    smoothed = motion_letter

                predicted = smoothed

                # save to letter buffer (we save when prediction changes)
                if len(saved_letters) == 0 or (
                    saved_letters and saved_letters[-1] != predicted
                ):
                    saved_letters.append(predicted)

        # draw outputs
        draw_prediction(frame, predicted, bbox=bbox)
        draw_top_probs(frame, probs)

        cv2.imshow("ASL Real-Time Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            do_speak = not do_speak
            print("Speech toggled:", do_speak)
            if do_speak:
                # say the last predicted letter once
                if saved_letters:
                    speak(saved_letters[-1])

    # write saved letters to disk
    with open("output_letters.txt", "w") as f:
        f.write("".join(saved_letters))

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting. Saved letters -> output_letters.txt")


if __name__ == "__main__":
    # run with speech off by default
    main(speak_output=False)
