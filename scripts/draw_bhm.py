import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

folder = "/home/daizhirui/D/Dev/erl_sddf/cmake-build-debug/src/erl_sdf_mapping"
sensor_position_file = "sensor_position.txt"
points_file = "points.txt"
# map_min_file = "map_min.txt"
# map_max_file = "map_max.txt"
centers_file = "centers.txt"
mins_file = "mins.txt"
maxs_file = "maxs.txt"

sensor_position = pd.read_csv(f"{folder}/{sensor_position_file}", header=None)
sensor_position = sensor_position.to_numpy().flatten()

points = pd.read_csv(f"{folder}/{points_file}", header=None)
points = points.to_numpy()  # (2, N)

# map_min = pd.read_csv(f"{folder}/{map_min_file}", header=None)
# map_min = map_min.to_numpy().flatten()
# map_max = pd.read_csv(f"{folder}/{map_max_file}", header=None)
# map_max = map_max.to_numpy().flatten()

centers = pd.read_csv(f"{folder}/{centers_file}", header=None)
centers = centers.to_numpy()  # (2, N)

mins = pd.read_csv(f"{folder}/{mins_file}", header=None)
mins = mins.to_numpy()  # (2, N)

maxs = pd.read_csv(f"{folder}/{maxs_file}", header=None)
maxs = maxs.to_numpy()  # (2, N)

plt.scatter(sensor_position[0], sensor_position[1], marker="o", color="r", label="Sensor Position")
plt.scatter(points[0], points[1], marker="x", color="b", label="Points")
plt.scatter(centers[0], centers[1], marker="*", color="y", label="Centers")
n = centers.shape[1]
for i in range(n):
    plt.gca().add_patch(
        patches.Rectangle(
            (mins[0, i], mins[1, i]),
            maxs[0, i] - mins[0, i],
            maxs[1, i] - mins[1, i],
            linewidth=1,
            edgecolor="g",
            facecolor="none",
        )
    )
# plt.gca().add_patch(
#     patches.Rectangle(
#         (map_min[0], map_min[1]),
#         map_max[0] - map_min[0],
#         map_max[1] - map_min[1],
#         linewidth=1,
#         edgecolor="g",
#         facecolor="none",
#         label="Map Area",
#     )
# )
plt.show()
