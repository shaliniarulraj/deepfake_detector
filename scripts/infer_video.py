import cv2
import torch
import argparse
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = models.resnet18(weights=None)
    in_f = model.fc.in_features
    model.fc = torch.nn.Linear(in_f, 2)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck["img_size"]

def sample_indices(n, k):
    if n <= 0:
        return []
    step = max(1, n // k)
    return list(range(0, n, step))[:k]

def detect_faces_dnn(net, frame, conf_thresh=0.5, min_face_size=30):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            bw, bh = x2 - x1, y2 - y1
            if bw >= min_face_size and bh >= min_face_size:
                boxes.append((x1, y1, bw, bh))
    return boxes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/best_resnet18.pt")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--face_model_dir", type=str, default="face_model",
                    help="Directory containing deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, img_size = load_model(args.ckpt, device)
    if args.img_size:
        img_size = args.img_size

    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Load DNN face detector
    proto_path = os.path.join(args.face_model_dir, "deploy.prototxt")
    model_path = os.path.join(args.face_model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Face detector model files not found in face_model_dir")
    face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    min_face_size = int(min(frame_w, frame_h) * 0.05)
    print(f"[INFO] Video resolution: {frame_w}x{frame_h}, Auto minSize: {min_face_size}px")

    idxs = sample_indices(n, args.samples)
    print(f"[INFO] Total frames: {n}, Sampling indices: {idxs}")

    probs = []

    with torch.no_grad():
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARN] Could not read frame {idx}")
                continue

            faces = detect_faces_dnn(face_net, frame, conf_thresh=0.5, min_face_size=min_face_size)
            print(f"[DEBUG] Frame {idx}: detected {len(faces)} faces")

            if len(faces) == 0:
                crop = frame
            else:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                pad = int(0.15 * max(w, h))
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                print(f"[WARN] Empty crop for frame {idx}")
                continue

            img = Image.fromarray(
                cv2.cvtColor(cv2.resize(crop, (img_size, img_size)), cv2.COLOR_BGR2RGB)
            )
            x_t = tfms(img).unsqueeze(0).to(device)
            p_fake = torch.softmax(model(x_t), dim=1)[0, 1].item()
            probs.append(p_fake)

    cap.release()

    if len(probs) == 0:
        print("No faces detected in sampled frames (even after fallback).")
    else:
        print(f"Mean fake probability: {float(np.mean(probs)):.4f}")
        print("Frame probs:", [round(p, 4) for p in probs])
