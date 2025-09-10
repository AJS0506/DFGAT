import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import HeteroGraphConv, TAGConv

"""
TAG : 단일 레이어에서 인접행렬에 따른 A0X , A1X , A2X, .. (k만큼) 을 각각 다른 Wk로 선형 변환 후 합침
-> 단일 레이어에서 다양한 k-hop 정보를 얻을 수 있음
"""

class TwoLayerSimpleHeteroTAG(nn.Module):
    def __init__(self, num_user_nodes, num_location_nodes, emb_dim, out1_dim, out2_dim, dataset):
        super(TwoLayerSimpleHeteroTAG, self).__init__()

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
        # TAGConv: k=2 (2-hop). 필요 시 k=1 or 3으로 조정 가능
        self.conv1 = HeteroGraphConv({
            'go': TAGConv(in_feats=self.emb_dim, out_feats=self.out1_dim, k=2),
            'back': TAGConv(in_feats=self.emb_dim, out_feats=self.out1_dim, k=2),
            # 'selftype1': TAGConv(self.emb_dim, self.out1_dim, k=2),
            # 'selftype2': TAGConv(self.emb_dim, self.out1_dim, k=2)
        }, aggregate='mean')

        # 첫 번째 레이어 출력에 대한 BatchNorm
        self.bn1_user = nn.BatchNorm1d(self.out1_dim)
        self.bn1_item = nn.BatchNorm1d(self.out1_dim)

        # ============ [ Layer 2 ] ============
        # 두 번째 HeteroGraphConv 레이어: out_dim -> out_dim
        self.conv2 = HeteroGraphConv({
            'go': TAGConv(in_feats=self.out1_dim, out_feats=self.out2_dim, k=2),
            'back': TAGConv(in_feats=self.out1_dim, out_feats=self.out2_dim, k=2),
            # 'selftype1': TAGConv(self.out1_dim, self.out2_dim, k=2),
            # 'selftype2': TAGConv(self.out1_dim, self.out2_dim, k=2)
        }, aggregate='mean')

        # 두 번째 레이어의 출력에 대한 BatchNorm
        self.bn2_user = nn.BatchNorm1d(self.out2_dim)
        self.bn2_item = nn.BatchNorm1d(self.out2_dim)

    def forward(self, g):
        # 초기 입력 임베딩
        inputs = {
            self.type1_name: self.type1_emb.weight,   # user 임베딩
            self.type2_name: self.type2_emb.weight    # item 임베딩
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
