import numpy as np
from pathlib import Path
from imageio import imread
# Calibration
fx, fy, cx1, cy = 3997.684, 3997.684, 1176.728, 1011.728
cx2 = 1307.839
baseline=193.001 # in millimeters

testF_folder = Path("demo-imgs/PipesH")

disp_path = f"demo_output/{testF_folder.name}.npy"
disp = np.load(disp_path)
image = imread(testF_folder / "im0.png")

# inverse-project
depth = (fx * baseline) / (-disp + (cx2 - cx1))
H, W = depth.shape
xx, yy = np.meshgrid(np.arange(W), np.arange(H))
points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth

mask = np.ones((H, W), dtype=bool)

# Remove flying points
mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False

points = points_grid.transpose(1,2,0)[mask]
colors = image[mask].astype(np.float64) / 255

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d



NUM_POINTS_TO_DRAW = 100000

subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW,), replace=False)
points_subset = points[subset]
colors_subset = colors[subset]

x, y, z = points_subset.T

fig=plt.figure(figsize=(8, 8))

ax = axes3d.Axes3D(fig)
ax.view_init(elev=12., azim=72)
ax.scatter(x, -z, -y,
           cmap='viridis',
           c=colors_subset,
           s=0.5,
           linewidth=0,
           alpha=1,
           marker=".")

plt.title('Point Cloud')
ax.axis('scaled')  # {equal, scaled}
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('point_cloud.png')




