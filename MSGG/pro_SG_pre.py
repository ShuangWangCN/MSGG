import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from MSGG.pro_GRU_channel import GRU_channel
from sklearn.metrics import mean_squared_error

class SG_pre(nn.Module):
    def __init__(self,vocabLabel,hidden_size):
        super(SG_pre, self).__init__()
        self.vocabLabel=vocabLabel
        self.hidden_size=hidden_size
        self.gru_channel=GRU_channel( vocabLabel, hidden_size)
        self.layer1 = nn.Sequential(nn.Linear(128+3*int(hidden_size /2), int(hidden_size /2)),
                                    nn.BatchNorm1d(int(hidden_size / 2)), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(int(hidden_size / 2), int(hidden_size / 4)),
                                    nn.BatchNorm1d(int(hidden_size / 4)), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(int(hidden_size / 4), 1))
        self.fc1 = nn.Linear(128+3*int(hidden_size /2), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, SGraph_batch, label_mean, label_std):

        label_mean = label_mean
        label_std = label_std
        label_batch, xc_batch= self.gru_channel(SGraph_batch)

        xc = self.fc1(xc_batch)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        loss = F.mse_loss(out, label_batch)
        pre_score_numpy = out.to('cpu').data.numpy()
        label_numpy = label_batch.to('cpu').data.numpy()
        real_label=label_numpy*label_std+label_mean
        real_pred_score=pre_score_numpy*label_std+label_mean
        mse=mean_squared_error(real_label, real_pred_score)
        rmse = np.sqrt(mse)
        return loss,rmse







