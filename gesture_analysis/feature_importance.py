import cv2
import numpy as np
import pandas as pd
import librosa
import mediapipe as mp
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────────
SINGERS = {
    "Anjana (Nagaraja)": {
      "video":  "female-vocals-front.mp4",
      "audio":  "female-vocals.wav"
    },
    "Amita (Krishnan)": {
      "video":  "female2-vocals.mp4",
      "audio":  "female2-vocals.wav"
    },
    "Prasanna (Soundararajan)": {
      "video":  "vocals-front.mp4",
      "audio":  "vocals.wav"
    },
    "Prashant (Krishnamoorthy)": {
      "video":  "male2-vocals.mp4",
      "audio":  "male2-vocals.wav"
    },
    "Salem (Shriram)": {
        "video": "validation.mp4",
        "audio": "validation.wav"
    }
}

mp_pose   = mp.solutions.pose
mp_face   = mp.solutions.face_mesh

def extract_note_df(video_path, audio_path):
    pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,           # highest‐quality pose model
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
    face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS)
    recs = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t   = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        p = pose.process(rgb)
        if not p.pose_landmarks:
            continue
        lm = p.pose_landmarks.landmark

        # Pose features
        sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                       lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        wr = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST]   .x,
                       lm[mp_pose.PoseLandmark.RIGHT_WRIST]   .y])
        hand_h = sh[1] - wr[1]

        # torso‐lean
        shL = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        shR = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        sh_mid = (shL + shR) / 2
        hpL    = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP].x,
                           lm[mp_pose.PoseLandmark.LEFT_HIP].y])
        hpR    = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
                           lm[mp_pose.PoseLandmark.RIGHT_HIP].y])
        hip_mid = (hpL + hpR) / 2
        torso_lean = sh_mid[1] - hip_mid[1]

        # elbow angle
        e = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
        v1 = sh_mid - e
        v2 = wr     - e
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
        elbow_ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        # head tilt
        nose      = np.array([lm[mp_pose.PoseLandmark.NOSE].x,
                              lm[mp_pose.PoseLandmark.NOSE].y])
        head_tilt = sh_mid[1] - nose[1]

        # FaceMesh features
        f_res = face_mesh.process(rgb)
        jaw_open = eye_open = mouth_w = np.nan
        if f_res.multi_face_landmarks:
            fl = f_res.multi_face_landmarks[0].landmark
            # jaw
            top = np.array([fl[13].x, fl[13].y])
            bot = np.array([fl[14].x, fl[14].y])
            jaw_open = np.linalg.norm(top - bot)
            # eye
            up = np.array([fl[159].x, fl[159].y])
            lo = np.array([fl[145].x, fl[145].y])
            eye_open = np.linalg.norm(up - lo)
            # mouth width
            l = np.array([fl[61].x, fl[61].y])
            r = np.array([fl[291].x, fl[291].y])
            mouth_w = np.linalg.norm(l - r)

        recs.append((t, hand_h, torso_lean, elbow_ang, head_tilt,
                     jaw_open, eye_open, mouth_w))
    cap.release()

    cols = ["time","hand_h","torso_lean","elbow_ang","head_tilt",
            "jaw_open","eye_open","mouth_w"]
    pose_df = pd.DataFrame(recs, columns=cols)
    pose_df[['jaw_open','eye_open','mouth_w']] = (
    pose_df[['jaw_open','eye_open','mouth_w']]
      .ffill().bfill()                                               # carry last good value forward/backward
      .fillna(pose_df[['jaw_open','eye_open','mouth_w']].median()) # fallback to median if still NaN
)
    # load & crop audio
    sr, y_int = wavfile.read(audio_path)
    if y_int.ndim>1:
        y_int = y_int.mean(axis=1)
    if np.issubdtype(y_int.dtype, np.integer):
        y = y_int.astype(np.float32) / np.iinfo(y_int.dtype).max
    else:
        y = y_int.astype(np.float32)
    video_dur = pose_df.time.max()
    y = y[:int(video_dur * sr)]

    # F0 + onsets
    hop    = int(sr / fps)
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=1500, sr=sr, hop_length=hop)
    onsets   = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    otimes   = librosa.frames_to_time(onsets, sr=sr, hop_length=hop)

    # merge frames→F0
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    audio_df = pd.DataFrame({"time": times, "f0": f0})
    df       = pd.merge_asof(pose_df, audio_df, on="time",
                             direction="nearest", tolerance=1/fps).dropna(subset=["f0"])

    # aggregate per‐note
    notes = []
    for i in range(len(otimes)-1):
        seg = df[(df.time >= otimes[i]) & (df.time < otimes[i+1])]
        if len(seg) < 5:
            continue
        means = seg[cols[1:]].mean()
        notes.append(tuple(means.values) + (seg.f0.mean(),))
    note_cols = cols[1:] + ["f0_mean"]
    notes_df  = pd.DataFrame(notes, columns=note_cols)

    # 1) drop features >90% missing
    thresh = len(notes_df) * 0.10
    notes_df = notes_df.dropna(axis=1, thresh=thresh)

    # 2) impute remaining NaNs by forward/backfill then median
    notes_df = notes_df.fillna(method="ffill").fillna(method="bfill")
    notes_df = notes_df.fillna(notes_df.median())

    # label
    notes_df["label"] = (notes_df.f0_mean > notes_df.f0_mean.median()).astype(int)
    return notes_df

# ─── MAIN: train + collect importances ─────────────────────────────────
all_imps = {}
for singer, paths in SINGERS.items():
    print(f"\n==== {singer.upper()} ====")
    nd = extract_note_df(paths["video"], paths["audio"])
    feats = [c for c in nd.columns if c not in ("f0_mean","label")]
    X, y  = nd[feats].values, nd["label"].values

    # drop zero‐var
    sel   = VarianceThreshold()
    X_sel = sel.fit_transform(X)
    kept  = np.array(feats)[sel.get_support()]

    rf    = RandomForestClassifier(n_estimators=200,
                                   max_depth=7,
                                   random_state=0)
    rf.fit(X_sel, y)

    imps  = dict(zip(kept, rf.feature_importances_))
    all_imps[singer] = imps

# compile & display
imp_df = pd.DataFrame(all_imps).T.fillna(0)
print("\nFeature importances across singers:")
print(imp_df)

imp_df.plot(kind="bar", figsize=(8,5),
            title="RandomForest importances by singer")
plt.ylabel("importance")
plt.tight_layout()
plt.show()