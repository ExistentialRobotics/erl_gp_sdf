import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


map_min = [-3.5, -5]
map_max = [-1, -2.5]
sensor_position = [12.254099, -10.823638]
point = [-2.8644285, -1.3291464]
sample = [-1.0000247, -2.499999]

fig, ax = plt.subplots()
ax.add_patch(
    patches.Rectangle(
        np.array(map_min),
        map_max[0] - map_min[0],
        map_max[1] - map_min[1],
        fill=False,
        color="blue",
        linewidth=2,
    )
)
plt.plot([sensor_position[0], point[0]], [sensor_position[1], point[1]], "r-")
plt.scatter(sensor_position[0], sensor_position[1], marker="o", color="r", label="Sensor Origin")
plt.scatter(point[0], point[1], marker="*", color="g", label="Sensor Points")
plt.scatter(sample[0], sample[1], marker="x", color="b", label="Sample Point")
plt.legend()
plt.axis("equal")
plt.show()
