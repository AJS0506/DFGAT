import torch.nn as nn
import dgl
from dgl.nn import HeteroGraphConv, SGConv

class TwoLayerSimpleHeteroSGC(nn.Module):
    def __init__(self, num_user_nodes, num_location_nodes, emb_dim, out1_dim, out2_dim, dataset):
        super(TwoLayerSimpleHeteroSGC, self).__init__()

        self.type1_name = "user"
        self.type2_name = "item"

        self.emb_dim = emb_dim
        self.out1_dim = out1_dim
        self.out2_dim = out2_dim

        # 사용자/아이템(혹은 location) 임베딩
        self.type1_emb = nn.Embedding(num_user_nodes, emb_dim)
        self.type2_emb = nn.Embedding(num_location_nodes, emb_dim)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.SiLU()

        # ============ [ Layer 1 ] ============
        # SGConv: (emb_dim -> out1_dim)
        # k=1 (1-hop), cached=False (캐싱 비활성), 기타 기본옵션
        # HeteroGraphConv의 aggregate='mean' 유지
        self.conv1 = HeteroGraphConv({
            'go':        SGConv(in_feats=self.emb_dim, out_feats=self.out1_dim, k=1),
            'back':      SGConv(in_feats=self.emb_dim, out_feats=self.out1_dim, k=1),
            # 'selftype1': SGConv(in_feats=self.emb_dim, out_feats=self.out1_dim, k=1),
            # 'selftype2': SGConv(in_feats=self.emb_dim, out_feats=self.out1_dim, k=1)
        }, aggregate='mean')

        # 첫 번째 레이어 출력에 대한 BatchNorm
        self.bn1_user = nn.BatchNorm1d(self.out1_dim)
        self.bn1_item = nn.BatchNorm1d(self.out1_dim)

        # ============ [ Layer 2 ] ============
        # SGConv: (out1_dim -> out2_dim)
        self.conv2 = HeteroGraphConv({
            'go':        SGConv(in_feats=self.out1_dim, out_feats=self.out2_dim, k=1),
            'back':      SGConv(in_feats=self.out1_dim, out_feats=self.out2_dim, k=1),
            # 'selftype1': SGConv(in_feats=self.out1_dim, out_feats=self.out2_dim, k=1),
            # 'selftype2': SGConv(in_feats=self.out1_dim, out_feats=self.out2_dim, k=1)
        }, aggregate='mean')

        # 두 번째 레이어 출력에 대한 BatchNorm
        self.bn2_user = nn.BatchNorm1d(self.out2_dim)
        self.bn2_item = nn.BatchNorm1d(self.out2_dim)

    def forward(self, g):
        # ===== 초기 입력 임베딩 =====
        inputs = {
            self.type1_name: self.type1_emb.weight,  # user 임베딩
            self.type2_name: self.type2_emb.weight   # item/location 임베딩
        }

        # ============ [ Layer 1 Forward ] ============
        h = self.conv1(g, inputs)

        # user 타입 노드
        h[self.type1_name] = self.bn1_user(h[self.type1_name])
        h[self.type1_name] = self.relu(h[self.type1_name])
        h[self.type1_name] = self.dropout(h[self.type1_name])

        # item 타입 노드
        h[self.type2_name] = self.bn1_item(h[self.type2_name])
        h[self.type2_name] = self.relu(h[self.type2_name])
        h[self.type2_name] = self.dropout(h[self.type2_name])

        # ============ [ Layer 2 Forward ] ============
        h = self.conv2(g, h)

        # user 타입 노드
        h[self.type1_name] = self.bn2_user(h[self.type1_name])
        h[self.type1_name] = self.relu(h[self.type1_name])
        h[self.type1_name] = self.dropout(h[self.type1_name])

        # item 타입 노드
        h[self.type2_name] = self.bn2_item(h[self.type2_name])
        h[self.type2_name] = self.relu(h[self.type2_name])
        h[self.type2_name] = self.dropout(h[self.type2_name])

        return h
