#!/usr/bin/env python3
# ============================================================================== #
# SimSiam - https://github.com/facebookresearch/simsiam
# Overlay -> Augment
# Extend 0.5 / 1 / 2 / 3 / 5 / 8
# Encoder ResNet-18 (32x32 size 3x3 CNN Kernel)
# SGD optimizer with weight decay 1e-4 and momentum 0.9, inital lr 0.025
# Powered by xiaolis@outlook.com
# ============================================================================== #
import os, math, time, random, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchvision.models import resnet18

from PIL.ImageFilter import GaussianBlur
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL.Image import blend
from random import shuffle

DEVICE = "cuda"
DATA_DIR = './data'
NUM_WORKERS = 4

# ============================================================================== #
torch.manual_seed(42)

class Simom:
    def __init__(self, data_ext=2):
        init_lr = 0.05 * 128 / 256
        criterion = nn.CosineSimilarity(dim=1).cuda(DEVICE)

        sim_trn_data = get_overlay_data(trn=True, shuf=False, ext=data_ext, batch_size=128)
        knn_trn_data = get_knn_data(train=True, aug=False, shuffle=True, batch_size=128)
        knn_val_data = get_knn_data(train=False, aug=False, shuffle=False, batch_size=100)

        model = SimSiam().to(DEVICE)
        optim_params = model.parameters()
        optimizer = torch.optim.SGD( params = optim_params, 
                                     lr = init_lr, 
                                     momentum = 0.9,
                                     weight_decay = 1e-4)

        from wandb import init ###
        monitor = init( project="simo", name=f"simo_d{data_ext}", config = {"version":"v0.3"}) ###
        for i in range(100):
            self._adjust_learning_rate(optimizer, init_lr, i)
            loss = train(sim_trn_data , model, criterion, optimizer, i)
            acc = downstream_knn(knn_trn_data, knn_val_data, model)
            print(f"Simo training epoch {i}/100 Similarity loss: {loss} Downstream Acc.: {acc*100:.4f}%")
            monitor.log({"acc": acc, "loss": loss}) ###
        monitor.finish() ###

    def _adjust_learning_rate(self, optimizer, init_lr, epoch):
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']: param_group['lr'] = init_lr
            else: param_group['lr'] = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / 100))

# ============================================================================== #
class GBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(GaussianBlur(radius=sigma))
        return x

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class Datax(Dataset):
    def __init__(self, raw, exs, shf=False):
        self.aug = TwoCropsTransform(self._aug_trans())
        self.exs = exs # output dataset size ratio 0.5/1/2
        self.raw = raw # input dataset
        self.dsz = len(raw) # dataset size
        if shf: self._shuffle()

    def _aug_trans(self):
        return T.Compose([ T.RandomResizedCrop(32, scale=(0.2, 1.)),
                           T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                           T.RandomGrayscale(p=0.2),
                           T.RandomApply([GBlur([.1, 2.])], p=0.5),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize( mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]) ])

    def __getitem__(self, idx):
        idx = idx%self.dsz
        return self.aug(self._ova(idx)), self.raw[idx][1]

    def __len__(self):
        return int(self.dsz*self.exs)

    def _shuffle(self):
        indices = list(range(self.dsz))
        shuffle(indices)
        self.raw = [self.raw[i] for i in indices]

    def _ova(self, idx):
        ixa, ixb = idx%self.dsz, self._ovb(idx)
        return blend(self.raw[ixa][0],self.raw[ixb][0],alpha=0.5)

    def _ovb(self, idx):
        for i in range(idx+1,self.dsz):
            if self.raw[idx][1]==self.raw[i][1]: return i
        return idx

def get_overlay_data(trn=True, shuf=False, ext=1, batch_size=128):
    raw_dataset = CIFAR10( root = DATA_DIR,
                           train = trn,
                           download = False )
    return DataLoader( dataset = Datax(raw=raw_dataset, exs=ext, shf=shuf), 
                       batch_size = batch_size, 
                       shuffle = shuf,
                       pin_memory = True,
                       drop_last = True,
                       num_workers = NUM_WORKERS )

def get_knn_data(train, aug, shuffle, batch_size):
    trans = T.Compose([ T.ToTensor(), 
                        T.Normalize( mean = (0.4914, 0.4822, 0.4465), 
                                     std = (0.2023, 0.1994, 0.2010)) ])
    if aug:
        trans = T.Compose([ T.RandomCrop(32, padding=4),
                            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply([GBlur([.1, 2.])], p=0.5),
                            T.RandomHorizontalFlip()
                            ] + [trans])
        trans = TwoCropsTransform(trans)
    return DataLoader( dataset = CIFAR10(root=DATA_DIR, train=train, download=False, transform=trans),
                       batch_size = batch_size,
                       shuffle = shuffle,
                       pin_memory = True,
                       drop_last = True,
                       num_workers = NUM_WORKERS )

# ============================================================================== #
class SimSiam(nn.Module):

    def __init__(self, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()
        # Resent-18 encoder
        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.fc = nn.Linear(pred_dim, dim)
        # 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential( nn.Linear(prev_dim, prev_dim, bias=False),
                                         nn.BatchNorm1d(prev_dim),
                                         nn.ReLU(inplace=True), # first layer
                                         nn.Linear(prev_dim, prev_dim, bias=False),
                                         nn.BatchNorm1d(prev_dim),
                                         nn.ReLU(inplace=True), # second layer
                                         self.encoder.fc,
                                         nn.BatchNorm1d(dim, affine=False) ) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
        # 2-layer predictor
        self.predictor = nn.Sequential( nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(pred_dim, dim) )
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

# ============================================================================== #
class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0,0,0,0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter('Loss', ':.4f')
    data_length = len(train_loader)

    model.train()
    end = time.time()
    cost = 0
    for i, (img, _) in enumerate(train_loader):
        img[0],img[1] = img[0].to(DEVICE), img[1].to(DEVICE)
        p1, p2, z1, z2 = model(x1=img[0], x2=img[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        losses.update(loss.item(), img[0].size(0))

        optimizer.zero_grad() # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if i+1 < data_length: continue
        cost += loss
    return cost/len(train_loader)

# ============================================================================== #
def downstream_knn(trn_data, test_data, model):
    trn_features, trn_labels = [], []
    for imgs, labs in trn_data:
        with torch.no_grad():
            features = model.encoder(imgs.to(DEVICE))
            trn_features.append(features)
            trn_labels.append(labs)
    trn_features = torch.cat(trn_features, dim=0)
    trn_labels = torch.cat(trn_labels, dim=0)

    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    knn_classifier.fit(trn_features.cpu().numpy(), trn_labels.cpu().numpy())

    test_features, test_labels = [], []
    for imgs, labs in test_data:
        with torch.no_grad():
            features = model.encoder(imgs.to(DEVICE))
            test_features.append(features)
            test_labels.append(labs)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    predicted_labels = knn_classifier.predict(test_features.cpu().numpy())
    accuracy = accuracy_score(test_labels.numpy(), predicted_labels)
    return accuracy

# ============================================================================== #
if __name__ == '__main__':
    Simom(0.5)
    Simom(1)
    Simom(2)
    Simom(3)
    Simom(5)
    Simom(8)
