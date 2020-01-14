import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from MSGG.GRU_channel import GRU_channel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

class SG_pre(nn.Module):
    def __init__(self,vocabLabel,hidden_size,task):
        super(SG_pre, self).__init__()
        self.vocabLabel=vocabLabel
        self.hidden_size=hidden_size
        self.gru_channel=GRU_channel( vocabLabel, hidden_size)
        self.layer1 = nn.Sequential(nn.Linear(3*int(hidden_size / 2), int(hidden_size /2)),nn.BatchNorm1d(int(hidden_size / 2)), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(int(hidden_size / 2), int(hidden_size / 4)),nn.BatchNorm1d(int(hidden_size / 4)), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(int(hidden_size / 4), 1))
        if task=='classification':
            self.class_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.30705128205128207))
    def forward(self, SGraph_batch, label_mean, label_std,task):
        if task=='regression':
            label_mean = label_mean
            label_std = label_std
            label_batch, TreeVec_batch, node_vec_tree_batch, edgeList_batch = self.gru_channel(SGraph_batch)
            graph_batch_vec = torch.cat([TreeVec_batch, node_vec_tree_batch,edgeList_batch], dim=1)
            pred_score = self.layer1(graph_batch_vec)
            pred_score = self.layer2(pred_score)
            pred_score = self.layer3(pred_score)
            loss =F.mse_loss(pred_score, label_batch)
            pre_score_numpy = pred_score.to('cpu').data.numpy()
            label_numpy = label_batch.to('cpu').data.numpy()
            real_label=label_numpy*label_std+label_mean
            real_pred_score=pre_score_numpy*label_std+label_mean
            mse=mean_squared_error(real_label, real_pred_score)
            rmse = np.sqrt(mse)
            return loss,rmse
        if task=='classification':
            label_batch, TreeVec_batch, node_vec_tree_batch, edgeList_batch = self.gru_channel(SGraph_batch)
            graph_batch_vec = torch.cat([TreeVec_batch, node_vec_tree_batch, edgeList_batch], dim=1)
            pred_score = self.layer1(graph_batch_vec)
            pred_score = self.layer2(pred_score)
            pred_score = self.layer3(pred_score)
            loss = self.class_loss(pred_score, label_batch)
            pre_score_numpy = pred_score.to('cpu').data.numpy()
            label_numpy = label_batch.to('cpu').data.numpy()
            roc_auc = roc_auc_score(label_numpy, pre_score_numpy)
            return loss, roc_auc








