import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, HeteroGraphConv


class TwoLayerSimpleHeteroGCN(nn.Module):
    def __init__(self, num_user_nodes, num_location_nodes, emb_dim, out1_dim, out2_dim, dataset):
        super(TwoLayerSimpleHeteroGCN, self).__init__()

        self.type1_name = "user"
        self.type2_name = "item"

        self.emb_dim = emb_dim

        self.out1_dim = out1_dim
        self.out2_dim = out2_dim

        self.type1_emb = nn.Embedding(num_user_nodes, emb_dim)
        self.type2_emb = nn.Embedding(num_location_nodes, emb_dim)

        self.dropout = nn.Dropout(0.2)

        self.relu = nn.SiLU()

        # ============ [ Layer 1 ] ============
        # 첫 번째 HeteroGraphConv 레이어: emb_dim -> out_dim
        self.conv1 = HeteroGraphConv({
            'go': GraphConv(self.emb_dim, self.out1_dim),
            'back': GraphConv(self.emb_dim, self.out1_dim),
            # 'selftype1': GraphConv(self.emb_dim, self.out1_dim),
            # 'selftype2': GraphConv(self.emb_dim, self.out1_dim)
        }, aggregate='mean')

        # 첫 번째 레이어의 출력에 대한 BatchNorm
        self.bn1_user = nn.BatchNorm1d(self.out1_dim)
        self.bn1_item = nn.BatchNorm1d(self.out1_dim)

        # ============ [ Layer 2 ] ============
        # 두 번째 HeteroGraphConv 레이어: out_dim -> out_dim
        self.conv2 = HeteroGraphConv({
            'go': GraphConv(self.out1_dim, self.out2_dim),
            'back': GraphConv(self.out1_dim, self.out2_dim),
            # 'selftype1': GraphConv(self.out1_dim, self.out2_dim),
            # 'selftype2': GraphConv(self.out1_dim, self.out2_dim)
        }, aggregate='mean')

        # 두 번째 레이어의 출력에 대한 BatchNorm
        self.bn2_user = nn.BatchNorm1d(self.out2_dim)
        self.bn2_item = nn.BatchNorm1d(self.out2_dim)


    def forward(self, g):
        # 초기 입력 임베딩
        inputs = {
            self.type1_name: self.type1_emb.weight,   # user 임베딩
            self.type2_name: self.type2_emb.weight    # location/movie 임베딩
        }

        # ============ [ Layer 1 Forward ] ============
        h = self.conv1(g, inputs)
        h[self.type1_name] = self.dropout(self.relu(self.bn1_user(h[self.type1_name])))
        h[self.type2_name] = self.dropout(self.relu(self.bn1_item(h[self.type2_name])))

        # ============ [ Layer 2 Forward ] ============
        h = self.conv2(g, h)
        h[self.type1_name] = self.dropout(self.relu(self.bn2_user(h[self.type1_name])))
        h[self.type2_name] = self.dropout(self.relu(self.bn2_item(h[self.type2_name])))

        return h



