import torch
import torch.nn.functional as F
import numpy as np
from math import log, pi
from torch import nn
from odefunc import ODEfunc, ODEnet
from cnf import CNF, SequentialFlow
from normalization import MovingBatchNorm1d

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2

def build_model(input_dim, hidden_dims, context_dim, num_blocks, conditional):
    def build_cnf():
        deffeq = ODEnet(hidden_dims= hidden_dims, 
            input_shape=(input_dim,), 
            context_dim= context_dim,
            layer_type= "concatsquash",
            nonlinearity="tanh"
            )
        odefunc = ODEfunc(deffeq = deffeq)
        cnf = CNF(odefunc= odefunc,
                    T = 0.5,
                    train_T= True,
                    conditional= conditional,
                    solver= "dopri5",
                    use_adjoint= True,
                    atol = 1e-5,
                    rtol= 1e-5)
        return cnf
    chain = [build_cnf() for _ in range(num_blocks)]
    bn_layers = [MovingBatchNorm1d(input_dim, bn_lag= 0, sync = False) for _ in range(num_blocks)]
    bn_chain = [MovingBatchNorm1d(input_dim, bn_lag= 0, sync = False)]
    for a,b in zip(chain, bn_layers):
        bn_chain.append(a)
        bn_chain.append(b)
    chain = bn_chain
    model = SequentialFlow(chain)
    return model

def create_point_cnf(input_dim, zdim, num_blocks):
    dims = tuple(512,512,512)
    model = build_model(input_dim,dims,zdim,num_blocks,True).cuda()
    return model

class Encoder(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv1d(3,128,1)
        self.conv2 = nn.Conv1d(128,128,1)
        self.conv3 = nn.Conv1d(128,256,1)
        self.conv4 = nn.Conv1d(256,512,1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1_m = nn.Linear(512,256)
        self.fc2_m = nn.Linear(256,128)
        self.fc3_m = nn.Linear(128,3)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        self.fc1_v = nn.Linear(512,256)
        self.fc2_v = nn.Linear(256,128)
        self.fc3_v = nn.Linear(128,3)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 3, keepdim = True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc1_v(x)))
        v = self.fc3_v(v)

        return m,v


class PointFlowModel(nn.Module):
    def __init__(self):
        super(PointFlowModel, self).__init__()
        self.encoder = Encoder()
        self.entropy_weight = 1
        self.recon_weight = 1
        self.prior_weight = 1
        self.input_dim = 3
        self.zdim = 128
        self.num_blocks = 1
        self.point_cnf = create_point_cnf(self.input_dim, self.zdim, self.num_blocks)
        self.latent_cnf = nn.Sequential()

    @staticmethod
    def sample_gaussian(size):
        y = torch.randn(*size).float()
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5* float(logvar.size(1)) * (1.+np.log(np.pi *2))
        ent = 0.5 * logvar.sum(dim=1, keepdim = False) + const
        return ent




    def forward(self, x, opt, step):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(x)
        z = self.reparameterize_gaussian(z_mu, z_sigma)
    
        # H[Q(z|X)]
        entropy = self.gaussian_entropy(z_sigma)

        # P(z)
        log_pz = torch.zeos(batch_size, 1).to(z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()
        y, delta_log_py = self.point_cnf(x, z_new, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim = True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        # loss
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        loss = entropy_loss + prior_loss + recon_loss
        loss.backward()
        opt.step()

        entropy_log = entropy.mean()
        recon = -log_px.mean()
        prior = -log_pz.mean()

        recon_nats = recon / float(x.size(1) * x.size(2))
        prior_nats = prior / float(self.zdim)

        return{
            'entropy':entropy_log.cpu().detach().item() if not isinstance(entropy_log, float) else entropy_log,
            'prior_nats':prior_nats,
            'recon_nats':recon_nats
        }
    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points):
        y =self.sample_gaussian((z.size(0), num_points, self.input_dim))
        x = self.create_point_cnf(y,z, reverse = True).view(*y.size())
        return y,x

    def get_parameters(self):
        return list(self.encoder.parameters()) + list(self.point_cnd.parameters()) + list(list(self.latent_cnf.parameters()))
