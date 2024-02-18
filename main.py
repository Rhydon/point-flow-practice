import numpy as np

path = "/rawdata3/ShapeNetCore.v2.PC15k/02691156/train/3b86245a5cd388ccf12b4513d8540d7c.npy"
pointCloud = np.load(path)

print(pointCloud.shape)

b = pointCloud[np.newaxis, :]
print(b.shape)