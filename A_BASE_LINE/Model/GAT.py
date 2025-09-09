import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, HeteroGraphConv, GATConv

class TwoLayerSimpleHeteroGAT(nn.Module):
    def __init__(self, 
                 num_user_nodes, 
                 num_location_nodes, 
                 emb_dim,       # 임베딩 차원
                 out1_dim,      # 첫 번째 레이어 출력 차원
                 out2_dim,      # 두 번째 레이어 출력 차원
                 dataset,
                 num_heads=3,   # GAT multi-head 수
                 allow_zero_in_degree = True,
                 feat_drop = 0.2,
                 attn_drop = 0.2,
                 negative_slope = 0.01,
                 residual = True,
                 dropout=0.2):

        super(TwoLayerSimpleHeteroGAT, self).__init__()
        
        self.type1_name = "user"
        self.type2_name = "item"
        
        self.emb_dim = emb_dim
        self.out1_dim = out1_dim
        self.out2_dim = out2_dim
        self.num_heads = num_heads

        self.type1_emb = nn.Embedding(num_user_nodes, emb_dim)
        self.type2_emb = nn.Embedding(num_location_nodes, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # ============ [ Layer 1 ] ============
        # GATConv(in_feats, out_feats, num_heads)는 (N, out_feats * num_heads)를 출력
        self.layer1 = HeteroGraphConv({
            'go':   GATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=num_heads,
                            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                            residual = residual),
            'back': GATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=num_heads,
                            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                            residual = residual),
        }, aggregate='mean') 
        
        self.nn1_user = nn.Linear(self.out1_dim * self.num_heads, self.out1_dim)  
        self.nn1_item = nn.Linear(self.out1_dim * self.num_heads, self.out1_dim)  

        self.bn1_user = nn.BatchNorm1d(out1_dim)
        self.bn1_item = nn.BatchNorm1d(out1_dim)

        # ============ [ Layer 2 ] ============
        self.layer2 = HeteroGraphConv({
            'go':   GATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=num_heads,
                            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                            residual = residual),
            'back': GATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=num_heads,
                            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                            residual = residual)
        }, aggregate='mean')

        self.nn2_user = nn.Linear(self.out2_dim * self.num_heads, self.out2_dim)  
        self.nn2_item = nn.Linear(self.out2_dim * self.num_heads, self.out2_dim)  

        self.bn2_user = nn.BatchNorm1d(out2_dim)
        self.bn2_item = nn.BatchNorm1d(out2_dim)

    def forward(self, g):
        # 초기 임베딩 세팅
        inputs = {
            self.type1_name: self.type1_emb.weight,   # shape: [num_users, emb_dim]
            self.type2_name: self.type2_emb.weight    # shape: [num_items, emb_dim]
        }

        # ============ [ Layer 1 Forward ] ============
        h_dict = self.layer1(g, inputs)
        # h_dict[user] shape: [num_users, out1_dim * num_heads]

        h_user = h_dict[self.type1_name]
        num_users = h_user.shape[0]
        #h_user = h_user.view(num_users, self.out1_dim, self.num_heads).mean(dim=2)  #-> GAT 결과를 합산
        h_user = h_user.view(num_users, self.out1_dim * self.num_heads)
        h_user = self.nn1_user(h_user)
        h_user = self.dropout(self.relu(self.bn1_user(h_user)))


        # item 쪽도 동일
        h_item = h_dict[self.type2_name]
        num_items = h_item.shape[0]
        #h_item = h_item.view(num_items, self.out1_dim, self.num_heads).mean(dim=2)
        h_item = h_item.view(num_items, self.out1_dim * self.num_heads)
        h_item = self.nn1_item(h_item)
        h_item = self.dropout(self.relu(self.bn1_item(h_item)))

        # Dictionary 형태로 묶어 Layer 2 입력으로
        h_dict = {
            self.type1_name: h_user,
            self.type2_name: h_item
        }

        # ============ [ Layer 2 Forward ] ============
        h2_dict = self.layer2(g, h_dict)
        # h2_dict[user] shape: [num_users, out2_dim * num_heads]
        h_user2 = h2_dict[self.type1_name]
        #h_user2 = h_user2.view(num_users, self.out2_dim, self.num_heads).mean(dim=2)
        h_user2 = h_user2.view(num_users, self.out2_dim * self.num_heads)
        h_user2 = self.nn2_user(h_user2)
        h_user2 = self.dropout(self.relu(self.bn2_user(h_user2)))

        h_item2 = h2_dict[self.type2_name]
        #h_item2 = h_item2.view(num_items, self.out2_dim, self.num_heads).mean(dim=2)
        h_item2 = h_item2.view(num_items, self.out2_dim * self.num_heads)
        h_item2 = self.nn2_item(h_item2)
        h_item2 = self.dropout(self.relu(self.bn2_item(h_item2)))

        return {
            self.type1_name: h_user2,
            self.type2_name: h_item2
        }


