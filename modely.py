import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tempfile
import requests
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Config
VIDEO_URL = "https://image.ihuoli.com/club/video/weixin/video_90732_1673839511.mp4"  # <- your URL
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
INPUT_SIZE = 192      # 192 for Lightning (faster), 256 for Thunder
CONF_THRESH = 0.3
SAVE_OUTPUT = True
OUTPUT_PATH = "annotated.mp4"

# Joint Map
COCO = {
  "nose":0,"left_eye":1,"right_eye":2,"left_ear":3,"right_ear":4,
  "left_shoulder":5,"right_shoulder":6,"left_elbow":7,"right_elbow":8,
  "left_wrist":9,"right_wrist":10,"left_hip":11,"right_hip":12,
  "left_knee":13,"right_knee":14,"left_ankle":15,"right_ankle":16
}

# Load MoveNet
print("Loading MoveNet from TF Hub...")
movenet = hub.load(MODEL_URL).signatures["serving_default"]
print("Model loaded.")

def infer_keypoints_bgr(frame_bgr, size=INPUT_SIZE):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    inp = tf.convert_to_tensor(resized[None, ...], dtype=tf.int32)
    out = movenet(inp)["output_0"].numpy()[0, 0]  # (17,3): y, x, score
    return out

def draw_pose(frame_bgr, kpts, conf=CONF_THRESH):
    h, w = frame_bgr.shape[:2]
    edges = [
        (0,1),(0,2),(1,3),(2,4),
        (0,5),(0,6),
        (5,7),(7,9),
        (6,8),(8,10),
        (5,11),(6,12),
        (11,13),(13,15),
        (12,14),(14,16),
        (11,12)
    ]
    # keypoints
    for (y, x, c) in kpts:
        if c < conf:
            continue
        cv2.circle(frame_bgr, (int(x*w), int(y*h)), 3, (0, 255, 0), -1)
    # skeleton
    for a, b in edges:
        if kpts[a,2] >= conf and kpts[b,2] >= conf:
            axy = (int(kpts[a,1]*w), int(kpts[a,0]*h))
            bxy = (int(kpts[b,1]*w), int(kpts[b,0]*h))
            cv2.line(frame_bgr, axy, bxy, (0, 200, 255), 2)
    return frame_bgr

def open_video_source(url_or_path):
    cap = cv2.VideoCapture(url_or_path)
    if cap.isOpened():
        return cap, None  # opened directly
    # Fallback: download to temp file, then open
    print("Direct open failed. Downloading to a temporary file...")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    with requests.get(url_or_path, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video after download. Possible codec issue in your OpenCV build.")
    return cap, tmp_path

def load_or_build_template_sequence(
    source,                      # URL or local path to the template video
    cache_path="template_sequence.npy",
    overwrite=False,
    sample_stride=1,             # keep every Nth frame (e.g., 2 to halve)
    min_conf=0.0,                # drop frames whose mean kp confidence < min_conf
    verbose=True
):
    """
    Returns: np.ndarray of shape (T, 17, 3) where each row is (y,x,score) in [0,1].
    """
    if (not overwrite) and os.path.exists(cache_path):
        if verbose: print(f"[Template] Loading cached: {cache_path}")
        return np.load(cache_path)

    if verbose: print(f"[Template] Extracting from: {source}")
    cap, tmp_path = open_video_source(source)

    seq = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_stride != 0:
            idx += 1
            continue

        kpts = infer_keypoints_bgr(frame)  # (17,3) y,x,score
        if min_conf > 0.0:
            mean_conf = float(np.mean(kpts[:, 2]))
            if mean_conf < min_conf:
                idx += 1
                continue

        seq.append(kpts)
        idx += 1

    cap.release()

    if tmp_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    seq = np.array(seq, dtype=np.float32)
    if seq.size == 0:
        raise RuntimeError("No frames extracted for template sequence. Check source or thresholds.")

    np.save(cache_path, seq)
    if verbose:
        print(f"[Template] Saved {seq.shape[0]} frames to: {cache_path}")
    return seq

# Normalizing pose input
def _mid(pts, a, b): return (pts[a] + pts[b]) / 2.0

def normalize_pose_2d(kpts_17x3):
    pts = kpts_17x3[:, :2].copy()
    # translate to pelvis midpoint
    origin = _mid(pts, COCO["left_hip"], COCO["right_hip"])
    pts -= origin
    # scale by torso length (mid-shoulder to origin)
    mid_sh = _mid(pts, COCO["left_shoulder"], COCO["right_shoulder"])
    torso = np.linalg.norm(mid_sh) + 1e-8
    pts /= torso
    return pts  # shape (17,2)

def joint_angle_deg(p, q, r):
    v1, v2 = p - q, r - q
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return np.nan
    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

ANGLE_TRIPLES = [
    ("left_shoulder","left_elbow","left_wrist"),
    ("right_shoulder","right_elbow","right_wrist"),
    ("left_hip","left_knee","left_ankle"),
    ("right_hip","right_knee","right_ankle"),
    ("nose","left_shoulder","left_elbow"),
    ("nose","right_shoulder","right_elbow"),
    ("left_shoulder","left_hip","left_knee"),
    ("right_shoulder","right_hip","right_knee"),
]

def limb_length(a, b, P): return np.linalg.norm(P[COCO[a]] - P[COCO[b]])

def pose_features(kpts_17x3):
    P = normalize_pose_2d(kpts_17x3)
    S = kpts_17x3[:, 2]  # confidences
    # angles
    angs = []
    for a,b,c in ANGLE_TRIPLES:
        ang = joint_angle_deg(P[COCO[a]], P[COCO[b]], P[COCO[c]])
        angs.append(ang)
    # limb ratios (scale-invariant)
    ua = limb_length("left_shoulder","left_elbow", P)
    fa = limb_length("left_elbow","left_wrist", P)
    ub = limb_length("right_shoulder","right_elbow", P)
    fb = limb_length("right_elbow","right_wrist", P)
    ta = limb_length("left_hip","left_knee", P)
    sa = limb_length("left_knee","left_ankle", P)
    tb = limb_length("right_hip","right_knee", P)
    sb = limb_length("right_knee","right_ankle", P)

    ratios = [
        fa/(ua+1e-6), fb/(ub+1e-6),
        sa/(ta+1e-6), sb/(tb+1e-6),
        (ua+ub)/(ta+tb+1e-6)  # upper-limb vs lower-limb
    ]

    feat = np.array(angs + ratios, dtype=np.float32)  # shape ~ (13,)
    # replace NaNs with per-frame mean or zero
    if np.isnan(feat).any():
        m = np.nanmean(feat)
        feat = np.nan_to_num(feat, nan=(0.0 if np.isnan(m) else m))
    return feat

def sequence_features(seq_17x3):
    seq_17x3 = np.asarray(seq_17x3)
    if seq_17x3.ndim != 3 or seq_17x3.shape[1:] != (17, 3):
        raise ValueError(f"Expected (T,17,3), got {seq_17x3.shape}")
    if seq_17x3.shape[0] == 0:
        raise ValueError("Empty sequence (T == 0)")
    F = np.stack([pose_features(k) for k in seq_17x3], axis=0)
    mu = F.mean(axis=0, keepdims=True)
    sigma = F.std(axis=0, keepdims=True) + 1e-6
    return (F - mu) / sigma

def dtw_score(user_seq, tmpl_seq, tau=2.0):
    # user_seq, tmpl_seq: arrays of shape (T,D)
    dist, _ = fastdtw(user_seq, tmpl_seq, dist=euclidean)
    # Normalize by average path length to reduce dependence on T
    path_len = (len(user_seq) + len(tmpl_seq)) / 2.0
    norm_cost = dist / max(path_len, 1.0)
    # Convert to score in [0,1]
    score = np.exp(-norm_cost / tau)  # smoother than linear
    return float(score), norm_cost

def dtw_scoring():
    # 1) Load sequences of keypoints
    template_seq = np.load("template_sequence.npy")      # (Tt,17,3)
    user_seq = np.load("user_sequence.npy")              # (Tu,17,3)

    # Optional: sanity checks
    for name, arr in [("template_seq", template_seq), ("user_seq", user_seq)]:
        print(f"[debug] {name} shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
        if arr.ndim != 3 or arr.shape[1:] != (17,3):
            raise ValueError(f"{name} must be (T,17,3), got {arr.shape}")

    # 2) Convert to features
    tmpl_feats = sequence_features(template_seq)         # (Tt,D)
    user_feats = sequence_features(user_seq)             # (Tu,D)

    # 3) DTW score
    score, norm_cost = dtw_score(user_feats, tmpl_feats) # [0,1], lower cost = better

    print(f"DTW score: {score:.3f} (norm_cost={norm_cost:.3f})")

def main():
    template_seq = load_or_build_template_sequence(
        VIDEO_URL,
        cache_path = "user_sequence.npy",
        overwrite = False,
        sample_stride = 1,  # or 2/3 to speed up
        min_conf = 0.2,  # filter low-confidence frames (optional)
        verbose = True
    )

    cap, tmp_path = open_video_source(VIDEO_URL)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"Opened video: {w}x{h} @ {fps:.2f} FPS")

    writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("Warning: Could not open VideoWriter. Disabling save.")
            writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kpts = infer_keypoints_bgr(frame)
        frame = draw_pose(frame, kpts)

        cv2.imshow("MoveNet (press q to quit)", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to: {OUTPUT_PATH}")
    cv2.destroyAllWindows()

    if tmp_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    dtw_scoring()

if __name__ == "__main__":
    main()
