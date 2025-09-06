import os, cv2, glob, argparse
from tqdm import tqdm

def sample_indices(n_frames, n_samples=12):
  if n_frames <= 0: return []
  step = max(1, n_frames // n_samples)
  return list(range(0, n_frames, step))[:n_samples]

def detect_largest_face(gray, face_cascade):
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
  if len(faces) == 0: return None
  # Pick largest face
  x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
  return x, y, w, h

def process_split(in_root, out_root, n_samples, size):
  os.makedirs(out_root, exist_ok=True)
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  for label in ["real", "fake"]:
    in_dir = os.path.join(in_root, label)
    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)
    video_paths = sorted(glob.glob(os.path.join(in_dir, "*.*")))
    for vp in tqdm(video_paths, desc=f"{os.path.basename(in_root)}:{label}"):
      cap = cv2.VideoCapture(vp)
      n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      idxs = sample_indices(n_frames, n_samples)
      base = os.path.splitext(os.path.basename(vp))[0]
      for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = detect_largest_face(gray, face_cascade)
        if box is None: continue
        x, y, w, h = box
        pad = int(0.15 * max(w, h))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue
        crop = cv2.resize(crop, (size, size))
        out_name = f"{base}_f{idx:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
      cap.release()

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--raw_root", type=str, default="data/raw")
  ap.add_argument("--out_root", type=str, default="data/frames")
  ap.add_argument("--samples_per_video", type=int, default=10)
  ap.add_argument("--face_size", type=int, default=160)
  args = ap.parse_args()
  for split in ["train", "val", "test"]:
    process_split(os.path.join(args.raw_root, split),
                  os.path.join(args.out_root, split),
                  args.samples_per_video,
                  args.face_size)
