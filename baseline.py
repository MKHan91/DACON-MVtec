
# import sys
# import os.path as osp
# import os
# sys.path.append(osp.join(os.getcwd(), "test"))


import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time
from torchsampler.imbalanced import ImbalancedDatasetSampler

device = torch.device('cuda')

train_png = sorted(glob('open/train/*.png'))
test_png = sorted(glob('open/test/*.png'))

# %%
train_y = pd.read_csv("open/train_df.csv")

train_labels = train_y["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]
# %%
def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (512, 512))
    return img

# %%
train_imgs = [img_load(m) for m in tqdm(train_png)]
test_imgs = [img_load(n) for n in tqdm(test_png)]


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]

        if self.mode=='train':
            augmentation = random.randint(0,2)
            if augmentation==1:
                img = img[::-1].copy()
            elif augmentation==2:
                img = img[:,::-1].copy()

        img = transforms.ToTensor()(img)
        if self.mode=='test':
            pass

        label = self.labels[idx]
        return img, label

    def get_labels(self):
        return self.labels
    

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)
        return x
    

batch_size = 32
epochs = 120

# Train
train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          sampler=ImbalancedDatasetSampler(train_dataset))

# Test
test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


# # ImbalancedSampler Debugging ------------------------------------
# invert_label_unique = {value:key for key, value in label_unique.items()}
# df = pd.read_csv("open/train_df.csv")

# for idx, batch in enumerate(train_loader):
#     # if idx == 1: break
#     x = torch.tensor(batch[0], dtype=torch.float32, device=device)
#     y = torch.tensor(batch[1], dtype=torch.long, device=device)
    
#     label_names = []
#     print(f"{'sampled batch label':<30} | {'Total Count'}")
#     print("-" * 50)
#     for sample in y:
#         sample = sample.item()
#         label_name = invert_label_unique[sample]
#         label_names.append(label_name)

#         sampled_label_cnt = len(df[df['label'] == label_name])
#         print(f"{label_name:<30} | {sampled_label_cnt:>10}")
#     print(label_names)
#     print(y)
# # ---------------------------------------------------------------------


model = Network().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


best=0
train_f1_list = []
for epoch in range(epochs):
    start=time.time()
    train_loss = 0
    train_pred=[]
    train_y=[]
    model.train()
    for batch in (train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = criterion(pred, y)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()/len(train_loader)
        train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        train_y += y.detach().cpu().numpy().tolist()


    train_f1 = score_function(train_y, train_pred)
    train_f1_list.append(train_f1)
    # print(train_f1_list)
    if max(train_f1_list) > train_f1_list[-1]:
        torch.save(model.state_dict(), "./model/"+str(epoch)+".pt")
    
    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')

# %%
