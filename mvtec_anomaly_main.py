import sys
import os
import os.path as osp
sys.path.append(osp.join(os.getcwd(), "test"))

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import cv2
import timm
import random
import time
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from torchsampler.imbalanced import ImbalancedDatasetSampler
from mvtec_anomaly_dataloader import mvtecDatasetPreprocess, mvtecDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)
        return x



def debugging_ImbalancedSampler(label_unique, train_loader, device):
    # ImbalancedSampler Debugging ------------------------------------
    invert_label_unique = {value:key for key, value in label_unique.items()}
    df = pd.read_csv("open/train_df.csv")

    for idx, sample_data in enumerate(train_loader):
        if idx == 1: break
        # print(sample_data)""
        x = torch.tensor(sample_data[0], dtype=torch.float32, device=device)
        y = torch.tensor(sample_data[1], dtype=torch.long, device=device)
        
        label_names = []
        print(f"{'sampled sample_data label':<30} | {'Total Count'}")
        print("-" * 50)
        for sample in y:
            sample = sample.item()
            label_name = invert_label_unique[sample]
            label_names.append(label_name)

            sampled_label_cnt = len(df[df['label'] == label_name])
            print(f"{label_name:<30} | {sampled_label_cnt:>10}")
        print(label_names)
        print(y)
    # ---------------------------------------------------------------------



def main():
    # 파라미터 셋팅
    batch_size = 32
    num_epochs = 120
    learning_rate = 1e-3
    folds = 5

    # GPU 셋팅
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{0}')
        print("Use GPU: {} for training".format(0))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    
    data_dir  = osp.join(osp.dirname(__file__), 'open')
    model_dir = osp.join(osp.dirname(__file__), 'experiments', 'models')
    log_dir   = osp.join(osp.dirname(__file__), 'experiments', 'logs')
    
    # 폴더 존재 점검
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    
    # 텐서보드 셋팅
    writer = SummaryWriter(log_dir)
    
    # 학습 데이터
    preprocessor = mvtecDatasetPreprocess(data_dir, mode='train')

    # # 테스트 데이터
    # test_dataset = mvtecDataset(data_dir, mode='test')
    # test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 교차검증
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state = 1)
    for fold, (indices_train, indices_valid) in enumerate(kfold.split(preprocessor.image_paths, preprocessor.labels)):
        subset_train_image_paths = [preprocessor.image_paths[idx] for idx in indices_train]
        subset_train_labels      = [preprocessor.labels[idx] for idx in indices_train]
        
        subset_valid_image_paths = [preprocessor.image_paths[idx] for idx in indices_valid]
        subset_valid_labels      = [preprocessor.labels[idx] for idx in indices_valid]
        
        train_subset = mvtecDataset(subset_train_image_paths, subset_train_labels)
        valid_subset = mvtecDataset(subset_valid_image_paths, subset_valid_labels)
        
        train_loader  = DataLoader(train_subset, batch_size=batch_size, 
                                   sampler=ImbalancedDatasetSampler(train_subset), 
                                   pin_memory=True, 
                                   num_workers=8)
        valid_loader  = DataLoader(valid_subset, batch_size=batch_size, 
                                   shuffle=False, 
                                   pin_memory=True, 
                                   num_workers=8)
        
        # 모델 정의
        model = Network().to(device)

        # 최적화
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
    
        # ------------------------------------ 학습 ------------------------------------ 
        total_train_steps   = torch.tensor(len(train_loader), dtype=torch.int32, device=device)
        total_valid_steps   = torch.tensor(len(valid_loader), dtype=torch.int32, device=device)
        for epoch in range(num_epochs):
            start = time.time()
            
            train_loss = torch.zeros(1,     device=device)
            model.train()
            for train_step, (sample_image, sample_label) in enumerate(train_loader):
                # if train_step > 1: break
                start = datetime.now()

                optimizer.zero_grad()
                sample_image = torch.tensor(sample_image, dtype=torch.float32, device=device)
                sample_label = torch.tensor(sample_label, dtype=torch.long,    device=device)
                
                with torch.cuda.amp.autocast():
                    pred = model(sample_image)
                    loss = criterion(pred, sample_label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += torch.div(loss, total_train_steps)
                
                """
                (a1 + a2 + ... + a100) / 100 = 평균
                a1/100 + a2/100 + .... + a100/100 = 평균
                """
                
                elapsed_time = datetime.now() - start
                if train_step % 30 == 0:
                    print(f"[TRAIN] Elapsed time: {elapsed_time} | Epoch: [{epoch + 1:>4}/{num_epochs}] | step: {train_step+1:>4}/{len(train_loader)} | Train Loss: {train_loss.tolist()[0]:4.4f}")
            
            # ------------------------------------------------------------------------------
            
            # ------------------------------------ 검증 ------------------------------------
            valid_loss    = torch.zeros(1,   device=device)
            valid_f1      = 0
            valid_f1_list = []
            model.eval()
            for sample_image, sample_label in valid_loader:
                start = datetime.now()
                
                sample_image = torch.tensor(sample_image, dtype=torch.float32, device=device)
                sample_label = torch.tensor(sample_label, dtype=torch.long,    device=device)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred = model(sample_image)
                        loss = criterion(pred, sample_label)
                
                valid_loss += torch.div(loss, total_valid_steps)
                
                valid_pred  = pred.argmax(1).detach().cpu().numpy().tolist()
                sample_label = sample_label.detach().cpu().numpy().tolist()

                valid_f1 += (f1_score(sample_label, valid_pred, average="macro") / len(valid_loader))
                valid_f1_list.append(valid_f1)
                
                
            if len(valid_f1_list) > 1 and (valid_f1 > max(valid_f1_list[:-1])):
                torch.save(model.state_dict(), osp.join(model_dir, f'fold_{fold:02d}-epoch_{epoch:03d}.pth'))
            
            elapsed_time = datetime.now() - start
            time_left = ((num_epochs - epoch+1) * elapsed_time.seconds) / 3600
            print(f"[VALID] Time Left: {time_left:4.2f} | Epoch: [{epoch + 1:>4}/{num_epochs}] | Valid Loss: {valid_loss.tolist()[0]:4.4f} | Valid F1: {valid_f1_list[-1]*100:.4f}%")
            
            # 텐서보드
            writer.add_scalar('Loss/Train Loss', train_loss.tolist()[0], global_step=epoch)
            writer.add_scalar('Loss/Valid Loss', valid_loss.tolist()[0], global_step=epoch)
            
if __name__ == "__main__":
    main()