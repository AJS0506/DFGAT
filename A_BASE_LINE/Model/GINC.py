import torch.nn as nn
import dgl
from dgl.nn import HeteroGraphConv, GINConv

class TwoLayerSimpleHeteroGINC(nn.Module):
    def __init__(self, num_user_nodes, num_location_nodes, emb_dim, out1_dim, out2_dim, dataset):
        super(TwoLayerSimpleHeteroGINC, self).__init__()

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
        # 첫 번째 HeteroGraphConv 레이어: emb_dim -> out1_dim
        # GINConv( apply_func=nn.Linear(...), aggregator_type='sum', ... )
        self.conv1 = HeteroGraphConv({
            'go': GINConv(
                apply_func=nn.Linear(self.emb_dim, self.out1_dim),
                aggregator_type='sum',
                init_eps=0,
                learn_eps=True,
                activation=None
            ),
            'back': GINConv(
                apply_func=nn.Linear(self.emb_dim, self.out1_dim),
                aggregator_type='sum',
                init_eps=0,
                learn_eps=True,
                activation=None
            )
            # 필요하다면 selftype1, selftype2 등 다른 edge type도 추가
        }, aggregate='mean')

        # 첫 번째 레이어의 출력에 대한 BatchNorm
        self.bn1_user = nn.BatchNorm1d(self.out1_dim)
        self.bn1_item = nn.BatchNorm1d(self.out1_dim)

        # ============ [ Layer 2 ] ============
        # 두 번째 HeteroGraphConv 레이어: out1_dim -> out2_dim
        self.conv2 = HeteroGraphConv({
            'go': GINConv(
                apply_func=nn.Linear(self.out1_dim, self.out2_dim),
                aggregator_type='sum',
                init_eps=0,
                learn_eps=True,
                activation=None
            ),
            'back': GINConv(
                apply_func=nn.Linear(self.out1_dim, self.out2_dim),
                aggregator_type='sum',
                init_eps=0,
                learn_eps=True,
                activation=None
            )
            # 필요하다면 selftype1, selftype2 등 다른 edge type도 추가
        }, aggregate='mean')

        # 두 번째 레이어의 출력에 대한 BatchNorm
        self.bn2_user = nn.BatchNorm1d(self.out2_dim)
        self.bn2_item = nn.BatchNorm1d(self.out2_dim)

    def forward(self, g):
        # 초기 입력 임베딩
        inputs = {
            self.type1_name: self.type1_emb.weight,  # user 임베딩
            self.type2_name: self.type2_emb.weight   # item 임베딩
        }

        # ============ [ Layer 1 Forward ] ============
        h = self.conv1(g, inputs)
        # user 노드
        h[self.type1_name] = self.bn1_user(h[self.type1_name])
        h[self.type1_name] = self.relu(h[self.type1_name])
        h[self.type1_name] = self.dropout(h[self.type1_name])
        # item 노드
        h[self.type2_name] = self.bn1_item(h[self.type2_name])
        h[self.type2_name] = self.relu(h[self.type2_name])
        h[self.type2_name] = self.dropout(h[self.type2_name])

        # ============ [ Layer 2 Forward ] ============
        h = self.conv2(g, h)
        # user 노드
        h[self.type1_name] = self.bn2_user(h[self.type1_name])
        h[self.type1_name] = self.relu(h[self.type1_name])
        h[self.type1_name] = self.dropout(h[self.type1_name])
        # item 노드
        h[self.type2_name] = self.bn2_item(h[self.type2_name])
        h[self.type2_name] = self.relu(h[self.type2_name])
        h[self.type2_name] = self.dropout(h[self.type2_name])

        return h
