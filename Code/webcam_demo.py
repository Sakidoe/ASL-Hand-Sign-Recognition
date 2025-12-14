# Live ASL letter prediction using our trained CNN + webcam.

import cv2
import numpy as np
import torch

from train_asl import SimpleCNN, index_to_letter, DEVICE


def preprocess_frame(frame):
    """
    Turns a webcam frame into something similar to the Kaggle dataset:
    - center crop
    - grayscale + blur
    - histogram equalization
    - threshold + invert
    - crop around the biggest contour (hopefully the hand)
    - resize to 28x28 and normalize
    """
    h, w, _ = frame.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame[top: top + side, left: left + side]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = 255 - thresh

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # If we find a contour, crop to the hand
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest)
        pad = 10
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w_box + pad, thresh.shape[1])
        y1 = min(y + h_box + pad, thresh.shape[0])
        hand_region = thresh[y0:y1, x0:x1]
    else:
        # If nothing shows up, just use the whole thing
        hand_region = thresh

    resized = cv2.resize(hand_region, (28, 28),
                         interpolation=cv2.INTER_AREA)

    img = resized.astype(np.float32) / 255.0
    img = img.reshape(1, 1, 28, 28)
    tensor = torch.tensor(img, dtype=torch.float32).to(DEVICE)

    return tensor, resized


def main():
    # Load the model we trained earlier
    model = SimpleCNN(num_classes=26)
    model.load_state_dict(torch.load("asl_cnn.pth",
                                     map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor, debug_28x28 = preprocess_frame(frame)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_letter = index_to_letter(pred_idx)
            pred_conf = probs[0, pred_idx].item()

        # Draw prediction on the video feed
        text = f"Pred: {pred_letter} ({pred_conf:.2f})"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Show the square where we're cropping from
        h, w, _ = frame.shape
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        cv2.rectangle(frame, (left, top),
                      (left + side, top + side),
                      (255, 0, 0), 2)

        cv2.imshow("ASL Webcam Demo", frame)

        # Also show what the model actually sees (scaled up)
        debug_large = cv2.resize(debug_28x28, (280, 280),
                                 interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Preprocessed 28x28", debug_large)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
