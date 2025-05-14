import pandas as pd
import matplotlib.pyplot as plt

# 1) Re‐construct your final importances dict, now with “salem” added:
all_imps = {
    "female1": {"hand_h":0.182491, "torso_lean":0.135004, "elbow_ang":0.127201, "head_tilt":0.555304, "jaw_open":0.000000, "eye_open":0.000000, "mouth_w":0.000000},
    "female2": {"hand_h":0.091648, "torso_lean":0.110893, "elbow_ang":0.095118, "head_tilt":0.297885, "jaw_open":0.141413, "eye_open":0.094478, "mouth_w":0.168566},
    "male1"  : {"hand_h":0.191004, "torso_lean":0.170067, "elbow_ang":0.216649, "head_tilt":0.422279, "jaw_open":0.000000, "eye_open":0.000000, "mouth_w":0.000000},
    "male2"  : {"hand_h":0.210096, "torso_lean":0.211104, "elbow_ang":0.152486, "head_tilt":0.426313, "jaw_open":0.000000, "eye_open":0.000000, "mouth_w":0.000000},
    "salem"  : {"hand_h":0.167157, "torso_lean":0.244993, "elbow_ang":0.139955, "head_tilt":0.447894, "jaw_open":0.000000, "eye_open":0.000000, "mouth_w":0.000000},
}

# 2) Build the DataFrame
imp_df = pd.DataFrame(all_imps).T.fillna(0)

# 3) Rename the rows to full names
rename_map = {
    "female1": "Amita",
    "female2": "Anjana",
    "male1"  : "Prasanna",
    "male2"  : "Prashant",
    "salem"  : "Salem"
}
imp_df.rename(index=rename_map, inplace=True)


fig, ax = plt.subplots(figsize=(10,6))
imp_df.plot(kind="bar", ax=ax, title="RandomForest Feature Importances by Singer")
ax.set_ylabel("Importance")
plt.xticks(rotation=45, ha="right")

# move legend to the right of the plot
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

# give the plot a little extra room on the right
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()