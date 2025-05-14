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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import VarianceThreshold

# ─── 0. Paths & setup ───────────────────────────────────────────────────
VIDEO_PATH = "/Users/anishnair/Special_Topics_AI/vocal_path/female2-vocals.mp4"
AUDIO_PATH = "/Users/anishnair/Special_Topics_AI/vocal_path/female2-vocals.wav"

mp_pose   = mp.solutions.pose
mp_face   = mp.solutions.face_mesh

# initialize models
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# ─── 1. Extract frame‑level gesture features ────────────────────────────
cap     = cv2.VideoCapture(VIDEO_PATH)
fps     = cap.get(cv2.CAP_PROP_FPS)
records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # pose landmarks
    pres = pose.process(rgb)
    if not pres.pose_landmarks:
        continue
    lm = pres.pose_landmarks.landmark

    # hand height
    sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                   lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
    wr = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                   lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
    hand_h = sh[1] - wr[1]

    # torso lean
    shL = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
    shR = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
    hipL = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x,
                     lm[mp_pose.PoseLandmark.LEFT_HIP].y])
    hipR = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
                     lm[mp_pose.PoseLandmark.RIGHT_HIP].y])
    torso_lean = ((shL+shR)/2)[1] - ((hipL+hipR)/2)[1]

    # elbow angle
    e = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                  lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
    v1 = sh - e
    v2 = wr - e
    cosang = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    elbow_ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    # head tilt
    nose = np.array([lm[mp_pose.PoseLandmark.NOSE].x,
                     lm[mp_pose.PoseLandmark.NOSE].y])
    head_tilt = torso_lean  # same vertical diff uses torso_lean vector

    # FaceMesh
    fres = face_mesh.process(rgb)
    jaw_open = eye_open = mouth_w = np.nan
    if fres.multi_face_landmarks:
        fl = fres.multi_face_landmarks[0].landmark
        top = np.array([fl[13].x, fl[13].y]); bot = np.array([fl[14].x, fl[14].y])
        jaw_open = np.linalg.norm(top-bot)
        up = np.array([fl[159].x, fl[159].y]); lo = np.array([fl[145].x, fl[145].y])
        eye_open = np.linalg.norm(up-lo)
        l = np.array([fl[61].x, fl[61].y]); r = np.array([fl[291].x, fl[291].y])
        mouth_w = np.linalg.norm(l-r)

    records.append((t, hand_h, torso_lean, elbow_ang, head_tilt,
                    jaw_open, eye_open, mouth_w))
cap.release()

cols = ["time","hand_h","torso_lean","elbow_ang",
        "head_tilt","jaw_open","eye_open","mouth_w"]
pose_df = pd.DataFrame(records, columns=cols)

# ─── 1b. Impute FaceMesh drop‑outs ─────────────────────────────────────
pose_df[['jaw_open','eye_open','mouth_w']] = (
    pose_df[['jaw_open','eye_open','mouth_w']]
      .ffill().bfill()
      .fillna(pose_df[['jaw_open','eye_open','mouth_w']].median())
)

# ─── 1c. Crop audio to match video duration ────────────────────────────
sr, y_int = wavfile.read(AUDIO_PATH)
if y_int.ndim>1:
    y_int = y_int.mean(axis=1)
if np.issubdtype(y_int.dtype, np.integer):
    y = y_int.astype(np.float32)/np.iinfo(y_int.dtype).max
else:
    y = y_int.astype(np.float32)
video_dur = pose_df.time.max()
print(f"Video duration: {video_dur:.1f}s; cropping audio to match")
y = y[:int(video_dur*sr)]

# ─── 2. F0 + onsets ────────────────────────────────────────────────────
hop = int(sr/fps)
f0,_,_ = librosa.pyin(y, fmin=75, fmax=1500, sr=sr, hop_length=hop)
t0 = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
otimes = librosa.frames_to_time(onsets, sr=sr, hop_length=hop)

# ─── 3. Merge & drop mismatches ─────────────────────────────────────────
audio_df = pd.DataFrame({"time":t0, "f0":f0})
df = pd.merge_asof(pose_df.sort_values("time"),
                   audio_df.sort_values("time"),
                   on="time", direction="nearest",
                   tolerance=1/fps).dropna(subset=["f0"])

# ─── 4. Note‑level aggregation ──────────────────────────────────────────
note_feats = []
for i in range(len(otimes)-1):
    seg = df[(df.time>=otimes[i])&(df.time<otimes[i+1])]
    if len(seg)<2:
        continue
    means = seg[cols[1:]].mean()
    f0m = seg.f0.mean()
    note_feats.append((*means.values, f0m))

note_cols = cols[1:]+["f0_mean"]
notes_df = pd.DataFrame(note_feats, columns=note_cols)
if notes_df.empty:
    raise RuntimeError("No valid notes extracted—check onset detection or alignment!")

# ─── 5. Label & prepare features ────────────────────────────────────────
notes_df["label"] = (notes_df.f0_mean>notes_df.f0_mean.median()).astype(int)
feats = [c for c in note_cols if c not in ("f0_mean")]
Xn, yn = notes_df[feats].values, notes_df.label.values

# ─── 6. Drop zero‑var & scale ──────────────────────────────────────────
sel = VarianceThreshold()
X_sel = sel.fit_transform(Xn)
kept = np.array(feats)[sel.get_support()]
X_s = StandardScaler().fit_transform(X_sel)

# ─── 7. Train & ROC for multiple classifiers ───────────────────────────
models = {
    "Logistic": LogisticRegression(max_iter=2000),
    "RF": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0),
    "SVM": SVC(kernel='rbf', probability=True),
    "GB": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=0),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=0)
}
mean_fpr = np.linspace(0,1,100)
tprs, aucs = {n:[] for n in models}, {n:[] for n in models}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for tr, te in cv.split(X_s, yn):
    Xtr, Xte = X_s[tr], X_s[te]
    ytr, yte = yn[tr], yn[te]
    for name, clf in models.items():
        prob = clf.fit(Xtr,ytr).predict_proba(Xte)[:,1]
        fpr,tpr,_ = roc_curve(yte, prob)
        tprs[name].append(np.interp(mean_fpr, fpr, tpr))
        tprs[name][-1][0]=0.0
        aucs[name].append(auc(fpr,tpr))

# ─── 8. Plot ROC ───────────────────────────────────────────────────────
plt.figure()
plt.plot([0,1],[0,1], 'k--', label='Chance')
for name in models:
    m_tpr = np.mean(tprs[name], axis=0)
    m_auc = np.mean(aucs[name]); s_auc = np.std(aucs[name])
    plt.plot(mean_fpr, m_tpr,
             label=f"{name} (AUC={m_auc:.2f}±{s_auc:.2f})")
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Hold-Out Singer: Amita Nagarajan')
plt.legend(loc='lower right'); plt.tight_layout(); plt.show()
