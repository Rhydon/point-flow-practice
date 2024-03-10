import numpy as np
import torch
from dataloader import get_train_file_name
from PointFlowModel import PointFlowModel

path = "/rawdata3/ShapeNetCore.v2.PC15k/02691156/train/3b86245a5cd388ccf12b4513d8540d7c.npy"
pointCloud = np.load(path)

print(pointCloud.shape)

b = pointCloud[np.newaxis, :]
print(b.shape[2])

source = "/rawdata3/ShapeNetCore.v2.PC15k"
flag = "airplane"

all_path = get_train_file_name(source, flag)

torch.device("cuda:0")
torch.cuda.set_device(0)

model = PointFlowModel()
model = model.cuda('cuda')

start_epoch = 0
optimizer = torch.optim.Adam(model.get_parameters())


