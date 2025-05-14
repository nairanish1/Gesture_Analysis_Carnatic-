# Fixed full script: male2_vocal_pose_estimation.py

import cv2
import numpy as np
# monkey‑patch for older librosa versions
if not hasattr(np, 'complex'):
    np.complex = complex

import pandas as pd
import matplotlib.pyplot as plt
import librosa
import mediapipe as mp
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# ─── 0. Paths & setup ───────────────────────────────────────────────────
VIDEO_PATH = "/Users/anishnair/Special_Topics_AI/vocal_path/male2-vocals.mp4"
AUDIO_PATH = "/Users/anishnair/Special_Topics_AI/vocal_path/male2-vocals.wav"

mp_pose   = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,           # highest‐quality pose model
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ─── 1. Extract frame-level gesture features ─────────────────────────────
cap     = cv2.VideoCapture(VIDEO_PATH)
fps     = cap.get(cv2.CAP_PROP_FPS)
records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    p_res = pose.process(rgb)
    if not p_res.pose_landmarks:
        continue
    lm = p_res.pose_landmarks.landmark

    # 1) hand height
    sh_y   = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    wr_y   = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
    hand_h = sh_y - wr_y

    # 2) torso lean
    sh_mid  = np.mean([
        [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
        [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    ], axis=0)
    hip_mid = np.mean([
        [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y],
        [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
    ], axis=0)
    torso_lean = sh_mid[1] - hip_mid[1]

    # 3) elbow angle
    e = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
    s = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
    w = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
    v1, v2 = s - e, w - e
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    elbow_ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    # 4) head tilt
    nose      = np.array([lm[mp_pose.PoseLandmark.NOSE].x, lm[mp_pose.PoseLandmark.NOSE].y])
    head_tilt = sh_mid[1] - nose[1]

    # 5) FaceMesh features
    f_res = face_mesh.process(rgb)
    jaw_open, eye_open, mouth_w = np.nan, np.nan, np.nan
    if f_res.multi_face_landmarks:
        fl = f_res.multi_face_landmarks[0].landmark
        # jaw open
        top = np.array([fl[13].x, fl[13].y])
        bot = np.array([fl[14].x, fl[14].y])
        jaw_open = np.linalg.norm(top - bot)
        # eye open (right)
        up  = np.array([fl[159].x, fl[159].y])
        lo  = np.array([fl[145].x, fl[145].y])
        eye_open = np.linalg.norm(up - lo)
        # mouth width
        lm_l = np.array([fl[61].x, fl[61].y])
        lm_r = np.array([fl[291].x, fl[291].y])
        mouth_w = np.linalg.norm(lm_l - lm_r)

    records.append((t, hand_h, torso_lean, elbow_ang, head_tilt, jaw_open, eye_open, mouth_w))

cap.release()
cols = ["time","hand_h","torso_lean","elbow_ang","head_tilt","jaw_open","eye_open","mouth_w"]
pose_df = pd.DataFrame(records, columns=cols)

# ─── 1b. Impute any FaceMesh dropouts ────────────────────────────────────
pose_df[['jaw_open','eye_open','mouth_w']] = (
    pose_df[['jaw_open','eye_open','mouth_w']]
      .ffill().bfill()                                               # carry last good value forward/backward
      .fillna(pose_df[['jaw_open','eye_open','mouth_w']].median()) # fallback to median if still NaN
)

# ─── 2. Audio: load + crop to video duration ─────────────────────────────
sr, y_int = wavfile.read(AUDIO_PATH)
if y_int.ndim > 1:
    y_int = y_int.mean(axis=1)
y = (y_int.astype(np.float32) / np.iinfo(y_int.dtype).max) \
    if np.issubdtype(y_int.dtype, np.integer) else y_int.astype(np.float32)

video_dur = pose_df.time.max()
print(f"Video duration: {video_dur:.1f}s; cropping audio to match")
y = y[:int(video_dur * sr)]

# ─── 2a. F0 contour ─────────────────────────────────────────────────────
hop = int(sr / fps)
f0, _, _ = librosa.pyin(y, fmin=75, fmax=1500, sr=sr, hop_length=hop)
t0       = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)

# ─── 2b. Onsets ─────────────────────────────────────────────────────────
onsets = librosa.onset.onset_detect(
    y=y, sr=sr, hop_length=hop,
    backtrack=True, delta=0.1,
    pre_max=20, post_max=20,
    pre_avg=100, post_avg=100
)
print("Detected onsets:", len(onsets))
otimes = librosa.frames_to_time(onsets, sr=sr, hop_length=hop)

# ─── 3. Merge & drop mismatches ─────────────────────────────────────────
audio_df = pd.DataFrame({"time": t0, "f0": f0})
df = pd.merge_asof(
    pose_df.sort_values("time"),
    audio_df.sort_values("time"),
    on="time", direction="nearest", tolerance=1/fps
).dropna(subset=["f0"])

# ─── 4. Note-level aggregation ─────────────────────────────────────────
note_feats = []
for i in range(len(otimes) - 1):
    start, end = otimes[i], otimes[i+1]
    seg = df[(df.time >= start) & (df.time < end)]
    if len(seg) < 5:
        continue
    means = seg[cols[1:]].mean()
    f0m   = seg.f0.mean()
    note_feats.append((*means.values, f0m))

note_cols = cols[1:] + ["f0_mean"]
notes_df  = pd.DataFrame(note_feats, columns=note_cols)
print("Notes extracted:", notes_df.shape[0])
if notes_df.empty:
    raise RuntimeError("No valid notes—check onsets!")

# fill occasional NaNs, drop all-NaN cols
notes_df.fillna(notes_df.median(), inplace=True)
notes_df.dropna(axis=1, how="all", inplace=True)
# ─── 5. Label high vs. low ──────────────────────────────────────────────
median_f0      = notes_df.f0_mean.median()
notes_df["label"] = (notes_df.f0_mean > median_f0).astype(int)

# ─── 6. Build feature list dynamically ─────────────────────────────────
feature_names = [c for c in cols[1:] if c in notes_df.columns]
if not feature_names:
    raise RuntimeError("No gesture features survive NaN-drop!")

# ─── 7. Prepare X & y ───────────────────────────────────────────────────
Xn = notes_df[feature_names].values
yn = notes_df["label"].values

# ─── 8. VarianceThreshold ──────────────────────────────────────────────
selector     = VarianceThreshold(threshold=0.0)
Xn_sel       = selector.fit_transform(Xn)
kept_indices = selector.get_support(indices=True)
kept_features = [feature_names[i] for i in kept_indices]
print("Kept features:", kept_features)
if not kept_features:
    raise RuntimeError("All features zero-variance!")

# ─── 9. Scale & classify ───────────────────────────────────────────────
Xn_s = StandardScaler().fit_transform(Xn_sel)

models = {
    "Logistic": LogisticRegression(max_iter=2000),
    "RF"      : RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    "SVM"     : SVC(kernel='rbf', probability=True, C=1.0, gamma="scale"),
    "GB"      : GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=0),
    "MLP"     : MLPClassifier(
                   hidden_layer_sizes=(50,25),
                   activation='relu',
                   solver='adam',
                   alpha=1e-4,
                   learning_rate_init=1e-3,
                   max_iter=500,
                   random_state=0
               )
}

mean_fpr = np.linspace(0, 1, 100)

# prepare per-model containers
tprs = {name: [] for name in models}
aucs = {name: [] for name in models}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, test_idx in cv.split(Xn_s, yn):
    Xtr, Xte = Xn_s[train_idx], Xn_s[test_idx]
    ytr, yte = yn[train_idx], yn[test_idx]

    for name, clf in models.items():
        probas = clf.fit(Xtr, ytr).predict_proba(Xte)[:,1]
        fpr, tpr, _ = roc_curve(yte, probas)
        tprs[name].append( np.interp(mean_fpr, fpr, tpr) )
        tprs[name][-1][0] = 0.0
        aucs[name].append( auc(fpr, tpr) )

# plot them all
plt.figure()
plt.plot([0,1],[0,1], 'k--', label="Chance")

for name in models:
    mean_tpr = np.mean(tprs[name], axis=0)
    mean_auc = np.mean(aucs[name])
    std_auc  = np.std(aucs[name])
    plt.plot(
        mean_fpr, mean_tpr,
        label=f"{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})"
    )

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Hold-Out Singer: Prashant Krishnamoorthy")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()