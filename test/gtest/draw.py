import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from io import StringIO

gazebo_bin_data = "/home/daizhirui/D/Dev/erl_sddf/src/erl_sdf_mapping/data/gazebo_train.dat"
# read bytes
with open(gazebo_bin_data, "rb") as f:
    data = f.read()
# convert bytes to numpy array
frames = []
i = 0
while i < len(data):
    numel = int.from_bytes(data[i : i + 4], "little")
    i += 4
    angles = np.frombuffer(data[i : i + numel * 8], dtype=np.float64, count=numel)
    i += numel * 8
    ranges = np.frombuffer(data[i : i + numel * 8], dtype=np.float64, count=numel)
    i += numel * 8
    numel = int.from_bytes(data[i : i + 8], "little")
    i += 8
    pose = np.frombuffer(data[i : i + numel * 8], dtype=np.float64, count=numel)
    i += numel * 8
    translation = np.array([pose[0], pose[1]])
    rotation = np.array(
        [
            [pose[2], pose[4]],
            [pose[3], pose[5]],
        ]
    )
    frames.append((angles, ranges, rotation, translation))


test_position = np.array([-0.0898721, -1.417363])
hs = 0.8 * 4
gps_str = """
GP position, distance, half_size, active, num_samples
-0.2 -1.4, 0.111488, 0.8, 0, 0
-0.2 -1.8, 0.39817, 0.8, 0, 0
-0.2   -1, 0.431648, 0.8, 0, 0
-0.6 -1.4, 0.510423, 0.8, 0, 0
-0.6 -1.8, 0.637685, 0.8, 0, 0
-0.6   -1, 0.659107, 0.8, 0, 0
-0.2 -2.2, 0.790347, 0.8, 0, 0
-0.2 -0.6, 0.824749, 0.8, 0, 0
 0.2 -0.6, 0.867242, 0.8, 0, 0
  -1 -1.4, 0.910294, 0.8, 0, 0
-0.6 -2.2, 0.934212, 0.8, 0, 0
-0.6 -0.6, 0.96349, 0.8, 0, 0
  -1 -1.8, 0.987291, 0.8, 0, 0
-1 -1, 1.00126, 0.8, 0, 0
 0.6 -0.6, 1.06958, 0.8, 0, 0
  -1 -2.2, 1.20036, 0.8, 0, 0
"""
gps = pd.read_csv(StringIO(gps_str), sep=", ")
pd.set_option("display.max_columns", None)
print(gps)

# Draw the map
plt.figure(figsize=(8, 8))
map_min = np.array([-2.5, -2.5])
map_max = np.array([2.5, 2.5])
for angles, ranges, rotation, translation in frames:
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    pts = np.array([ranges * cos_angles, ranges * sin_angles])
    pts = np.dot(rotation, pts) + translation[:, np.newaxis]
    plt.plot(pts[0, :], pts[1, :], c="k", alpha=0.2)
    map_min[0] = min(map_min[0], np.min(pts[0, :]))
    map_min[1] = min(map_min[1], np.min(pts[1, :]))
    map_max[0] = max(map_max[0], np.max(pts[0, :]))
    map_max[1] = max(map_max[1], np.max(pts[1, :]))
map_min -= 0.5
map_max += 0.5

for i in range(len(gps)):
    position = gps["GP position"][i].strip().split(" ")
    print(position)
    position = [float(x.strip()) for x in position if x.strip()]
    if gps["active"][i] == 1:
        rect = patches.Rectangle(
            (position[0] - hs, position[1] - hs),
            2 * hs,
            2 * hs,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    else:
        rect = patches.Rectangle(
            (position[0] - hs, position[1] - hs),
            2 * hs,
            2 * hs,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
    plt.gca().add_patch(rect)
    plt.scatter(position[0], position[1], c="g", s=10)

plt.scatter(test_position[0], test_position[1], c="y", s=10)
plt.xlim(map_min[0], map_max[0])
plt.ylim(map_min[1], map_max[1])
plt.show()
