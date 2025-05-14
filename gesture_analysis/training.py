#!/usr/bin/env python3
"""
Leave-One-Singer-Out experiment for 5 Carnatic vocalists.
Predict high/low-pitch notes from hand/body/face gestures.
"""

import cv2
import librosa
import mediapipe as mp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.spatial.distance import cosine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ─── CONFIG ─────────────────────────────────────────────────────────────
SINGERS = {
    "Anjana (Nagarajan)"       : {"video": "female-vocals-front.mp4", "audio": "female-vocals.wav"},
    "Amita (Krishnan)"         : {"video": "female2-vocals.mp4",      "audio": "female2-vocals.wav"},
    "Prasanna (Soundararajan)" : {"video": "vocals-front.mp4",        "audio": "vocals.wav"},
    "Prashant (Krishnamoorthy)": {"video": "male2-vocals.mp4",        "audio": "male2-vocals.wav"},
    "Salem (Shriram)"          : {"video": "validation.mp4",          "audio": "validation.wav"}
}

FEATURE_COLS     = ["hand_h", "torso_lean", "elbow_ang", "head_tilt", "jaw_open", "eye_open", "mouth_w"]
FRAME_SKIP       = 1     # process every 2nd frame
MIN_FRAMES_NOTE  = 5     # drop notes shorter than this many frames
MISS_COL_THRESH  = 0.10  # drop features >90% missing

mp_pose, mp_face = mp.solutions.pose, mp.solutions.face_mesh

# ─── HELPER: extract per-note features ─────────────────────────────────
def extract_note_df(video: str, audio: str) -> pd.DataFrame:
    pose = mp_pose.Pose(
        static_image_mode=True, model_complexity=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    face = mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) / FRAME_SKIP
    recs = []

    while True:
        # skip intermediate frames
        for _ in range(FRAME_SKIP - 1):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            break

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose landmarks
        p = pose.process(rgb)
        if not p.pose_landmarks:
            continue
        lm = p.pose_landmarks.landmark

        # 1) hand height
        sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                       lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        wr = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
        hand_h = sh[1] - wr[1]

        # 2) torso lean
        shL = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        shR = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        sh_mid = (shL + shR) / 2
        hpL = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x,
                        lm[mp_pose.PoseLandmark.LEFT_HIP].y])
        hpR = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
                        lm[mp_pose.PoseLandmark.RIGHT_HIP].y])
        hip_mid = (hpL + hpR) / 2
        torso_lean = sh_mid[1] - hip_mid[1]

        # 3) elbow angle
        e = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
        v1, v2 = sh_mid - e, wr - e
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        elbow_ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        # 4) head tilt
        nose = np.array([lm[mp_pose.PoseLandmark.NOSE].x,
                         lm[mp_pose.PoseLandmark.NOSE].y])
        head_tilt = sh_mid[1] - nose[1]

        # 5) FaceMesh features
        fm = face.process(rgb)
        jaw = eye = mouth = np.nan
        if fm.multi_face_landmarks:
            fl = fm.multi_face_landmarks[0].landmark
            jaw = np.linalg.norm(np.array([fl[13].x, fl[13].y]) - np.array([fl[14].x, fl[14].y]))
            eye = np.linalg.norm(np.array([fl[159].x, fl[159].y]) - np.array([fl[145].x, fl[145].y]))
            mouth = np.linalg.norm(np.array([fl[61].x, fl[61].y]) - np.array([fl[291].x, fl[291].y]))

        recs.append((t, hand_h, torso_lean, elbow_ang, head_tilt, jaw, eye, mouth))

    cap.release()

    # Build DataFrame & impute FaceMesh dropouts
    df = pd.DataFrame(recs, columns=["time"] + FEATURE_COLS)
    df[["jaw_open", "eye_open", "mouth_w"]] = (
        df[["jaw_open", "eye_open", "mouth_w"]]
        .ffill().bfill()
        .fillna(df[["jaw_open", "eye_open", "mouth_w"]].median())
    )

    # Load audio & compute pitch contour
    sr, yint = wavfile.read(audio)
    if yint.ndim > 1:
        yint = yint.mean(axis=1)
    y = yint.astype(np.float32)
    if np.issubdtype(yint.dtype, np.integer):
        y /= np.iinfo(yint.dtype).max

    y = y[: int(df.time.max() * sr)]
    hop = int(sr / fps)

    # pyin signature with keywords
    f0, _, _ = librosa.pyin(y, sr=sr, fmin=75, fmax=1500, hop_length=hop)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    otimes = librosa.frames_to_time(onsets, sr=sr, hop_length=hop)

    frame_t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    df = pd.merge_asof(
        df,
        pd.DataFrame({"time": frame_t, "f0": f0}),
        on="time", direction="nearest", tolerance=1 / fps
    ).dropna(subset=["f0"])

    # Aggregate per-note
    notes = []
    for i in range(len(otimes) - 1):
        seg = df[(df.time >= otimes[i]) & (df.time < otimes[i+1])]
        if len(seg) < MIN_FRAMES_NOTE:
            continue
        notes.append((*seg[FEATURE_COLS].mean().values, seg.f0.mean()))

    notes_df = pd.DataFrame(notes, columns=FEATURE_COLS + ["f0_mean"])
    notes_df.dropna(axis=1, thresh=len(notes_df) * MISS_COL_THRESH, inplace=True)
    notes_df.ffill(inplace=True)
    notes_df.fillna(notes_df.median(), inplace=True)
    notes_df["label"] = (notes_df.f0_mean > notes_df.f0_mean.median()).astype(int)

    return notes_df

# MODELS 
models = {
    "Logistic": LogisticRegression(max_iter=2000, solver="liblinear"),
    "RF"      : RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    "SVM"     : SVC(kernel="rbf", probability=True),
    "GB"      : GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0),
    "MLP"     : MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=0)
}

# LOOV EXPERIMENT
print("⇢ extracting note-level data (once per singer)…")
notes_by_singer = {
    name: extract_note_df(**paths)
    for name, paths in SINGERS.items()
}

results = []
salem_confusion = None

for holdout in SINGERS:
    print(f"⇢ LOOV – holding out: {holdout}")
    # combine all other singers for training
    df_tr = pd.concat(
        [df for name, df in notes_by_singer.items() if name != holdout],
        ignore_index=True
    )
    df_te = notes_by_singer[holdout].copy()

    # ensure all FEATURE_COLS exist in both
    df_tr = df_tr.reindex(columns=FEATURE_COLS + ["label"], fill_value=np.nan)
    df_te = df_te.reindex(columns=FEATURE_COLS + ["label"], fill_value=np.nan)

    # median impute using training medians
    train_meds = df_tr[FEATURE_COLS].median()
    df_tr[FEATURE_COLS]=df_tr[FEATURE_COLS].fillna(train_meds)
    df_te[FEATURE_COLS]=df_te[FEATURE_COLS].fillna(train_meds)

    Xtr, ytr = df_tr[FEATURE_COLS].values, df_tr["label"].values
    Xte, yte = df_te[FEATURE_COLS].values, df_te["label"].values

    # drop zero-variance & scale
    sel = VarianceThreshold()
    Xtr_sel = sel.fit_transform(Xtr)
    Xte_sel = sel.transform(Xte)
    sc = StandardScaler().fit(Xtr_sel)
    Xtr_s, Xte_s = sc.transform(Xtr_sel), sc.transform(Xte_sel)

    # compute FI-distance
    rf_tr = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0).fit(Xtr_s, ytr)
    rf_hd = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0).fit(Xte_s, yte)
    fi_dist = cosine(rf_tr.feature_importances_, rf_hd.feature_importances_)

    # evaluate each model
    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        prob = clf.predict_proba(Xte_s)[:, 1]
        results.append({
            "holdout": holdout,
            "model"  : name,
            "AUC"    : roc_auc_score(yte, prob),
            "ACC"    : accuracy_score(yte, clf.predict(Xte_s)),
            "FI_dist": fi_dist
        })

    # save confusion matrix for Salem
    if holdout == "Salem (Shriram)":
        cm = confusion_matrix(yte, rf_tr.predict(Xte_s), labels=[0, 1])
        salem_confusion = ConfusionMatrixDisplay(cm, display_labels=["Low", "High"])

# ─── PLOT RESULTS ───────────────────────────────────────────────────────
res_df=pd.DataFrame(results)

plt.figure(figsize=(6,4))
sns.barplot(data=res_df, x="model", y="AUC", ci="sd")
plt.title("LOOV Generalization AUC")
plt.ylim(0,1)
plt.tight_layout()

plt.figure(figsize=(4,4))
rf_res=res_df[res_df.model == "RF"]
plt.scatter(rf_res.FI_dist, rf_res.AUC)
for _, r in rf_res.iterrows():
    plt.text(r.FI_dist, r.AUC, r.holdout, fontsize=8)
plt.xlabel("Feature-Importance Cosine Distance")
plt.ylabel("RF LOOV AUC")
plt.tight_layout()

plt.figure(figsize=(4,4))
salem_confusion.plot(cmap="Blues")
plt.title("Salem (Shriram) – RF Confusion")
plt.tight_layout()

plt.show()
