import numpy as np
import torch
import time
from dataloader import get_train_file_name, Dataloader
from PointFlowModel import PointFlowModel
from utils import AverageValueMeter, apply_random_rotation

path = "/rawdata3/ShapeNetCore.v2.PC15k/02691156/train/3b86245a5cd388ccf12b4513d8540d7c.npy"
source = "/rawdata3/ShapeNetCore.v2.PC15k"
flag = "airplane"
batch_size = 16
epoches = 4000
log_frequency = 10
exp_decay_frequency = 1

def init_np_seed():
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)

torch.device("cuda:0")
torch.cuda.set_device(0)

model = PointFlowModel()
model = model.cuda('cuda')
optimizer = torch.optim.Adam(model.get_parameters())

train_dataset = Dataloader(source, flag, 'train')
test_dataset = Dataloader(source, flag, 'val')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0, 
    pin_memory = True, sampler = None, drop_last = True, worker_init_fn = init_np_seed)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False, num_workers = 0, 
    pin_memory = True, drop_last = True, worker_init_fn = init_np_seed)

def lambda_rule(ep):
    lr_l = 1.0 - max(0, ep - 0.5 * epoches) / float(0.5 * epoches)
    return lr_l
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

start_time = time.time()
entropy_avg_meter = AverageValueMeter()
latent_avg_meter = AverageValueMeter()
point_nats_avg_meter = AverageValueMeter()

for epoch in range(epoches):
    for bidx, data in enumerate(train_loader):
        index_batch = data['idx']
        train_batch = data['train_points']
        test_batch = data['test_points']
        step = bidx + len(train_loader) * epoch
        model.train()

        inputs = train_batch.cuda(torch.device('cuda'), non_blocking = True)
        out = model(inputs, optimizer, step)

        entropy = out['entropy']
        prior_nats = out['prior_nats']
        recon_nats = out['recon_nats']
        entropy_avg_meter.update(entropy)
        point_nats_avg_meter.update(prior_nats)
        latent_avg_meter.update(recon_nats)

        if step % log_frequency ==0:
            duration = time.time() - start_time
            start_time = time.time()
            print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Entropy %2.5f LatentNats %2.5f PointNats %2.5f"
                    % (epoch, bidx, len(train_loader), duration, entropy_avg_meter.avg,
                        latent_avg_meter.avg, point_nats_avg_meter.avg))
    if(epoch + 1) % exp_decay_frequency == 0:
        scheduler.step(epoch = epoch)
        


