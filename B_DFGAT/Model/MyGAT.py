import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from collections import defaultdict

from dgl.nn import HeteroGraphConv, GATConv, GraphConv
from .TimeEncodingGAT import TimeEncodingGATConv
from .PopEncodingGAT import PopEncodingGATConv


class DiffHeadGAT(nn.Module):
    def __init__(self, 
                 num_user_nodes : int, 
                 num_location_nodes : int, 
                 emb_dim : int,       # 임베딩 차원
                 out1_dim : int,      # 첫 번째 레이어 출력 차원
                 out2_dim : int,      # 두 번째 레이어 출력 차원
                 uid2ts : dict, 
                 uid2dg : dict,
                 mid2dg : dict,
                 device : str,

                 allow_zero_in_degree = True,
                 feat_drop = 0.2,
                 attn_drop = 0.2,
                 negative_slope = 0.01,
                 residual = True,
                 dropout=0.2):
        
        super(DiffHeadGAT, self).__init__()
        
        # ============ 유저별 정규화된 타임스탬프 딕셔너리 ============
        self.uid2ts = uid2ts
        self.uid2dg = uid2dg
        self.mid2dg = mid2dg

        # ============ 이종그래프 타입 정의 ============
        self.type1_name = "user"
        self.type2_name = "item"

        # ============ 각 타입별 특징 임베딩 ============ 
        self.emb_dim = emb_dim
        self.type1_emb = nn.Embedding(num_user_nodes, emb_dim)
        self.type2_emb = nn.Embedding(num_location_nodes, emb_dim)

        # ============ 각 레이어별 출력 차원 수 ============ 
        self.out1_dim = out1_dim
        self.out2_dim = out2_dim

        # ============ 각 타입 별 노드 수 ============ 
        self.num_user_nodes = num_user_nodes
        self.num_location_nodes = num_location_nodes

        # ============ 드롭아웃 레이어 ============ 
        self.dropout = nn.Dropout(dropout)

        # ============ 활성화함수 정의 ============ 
        self.relu = nn.SiLU()

        # ============ [ GAT Layer 1 ] ============
        # 첫 번째 HeteroGraphConv 레이어: emb_dim -> out_dim
        self.layer1_head1 = HeteroGraphConv({
            'go':   GATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual),

            'back': GATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual)
        }, aggregate='mean') 

        self.layer1_head2 = HeteroGraphConv({
            'go':   TimeEncodingGATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2ts = self.uid2ts, device = device),

            'back': TimeEncodingGATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2ts = self.uid2ts, device = device)
        }, aggregate='mean')  
 
        
        self.layer1_head3 = HeteroGraphConv({
            'go':   PopEncodingGATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2dg = self.uid2dg, mid2dg = self.mid2dg, device = device),

            'back': PopEncodingGATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2dg = self.uid2dg, mid2dg = self.mid2dg, device = device)
        }, aggregate='mean')  

        self.layer1_concatenator = nn.Linear(out1_dim * 3, out1_dim)

        # 첫 번째 레이어의 출력에 대한 BatchNorm
        self.bn1_user = nn.BatchNorm1d(self.out1_dim)
        self.bn1_item = nn.BatchNorm1d(self.out1_dim)

        # ============ [ Layer 2 ] ============
        self.layer2_head1 = HeteroGraphConv({
            'go':   GATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual),

            'back': GATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual)
        }, aggregate='mean')

        self.layer2_head2 = HeteroGraphConv({
            'go':   TimeEncodingGATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2ts = self.uid2ts, device = device),

            'back': TimeEncodingGATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2ts = self.uid2ts, device = device)
        }, aggregate='mean')

        self.layer2_head3 = HeteroGraphConv({
            'go':   PopEncodingGATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2dg = self.uid2dg, mid2dg = self.mid2dg, device = device),

            'back': PopEncodingGATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2dg = self.uid2dg, mid2dg = self.mid2dg, device = device)
        }, aggregate='mean')

        self.layer2_concatenator = nn.Linear(out2_dim * 3, out2_dim)

        # 두 번째 레이어의 출력에 대한 BatchNorm
        self.bn2_user = nn.BatchNorm1d(self.out2_dim)
        self.bn2_item = nn.BatchNorm1d(self.out2_dim)


    def forward(self, g):

        inputs = {
            self.type1_name: self.type1_emb.weight,  
            self.type2_name: self.type2_emb.weight   
        }

        # ======================
        # 1) Layer 1 Forward
        # ======================

        h1_1 = self.layer1_head1(g, inputs)  # {'user':(610,1,out1_dim), 'item':(9742,1,out1_dim)}
        h1_2 = self.layer1_head2(g, inputs)
        h1_3 = self.layer1_head3(g, inputs)

        # --- 유저 헤드 수행 ---
        user_h1_1 = h1_1['user'].squeeze(1)  # shape: (num_users, out1_dim)
        user_h1_2 = h1_2['user'].squeeze(1)
        user_h1_3 = h1_3['user'].squeeze(1)

        user_concat_1 = torch.cat([user_h1_1, user_h1_2, user_h1_3], dim=1)
        # shape: (num_users, out1_dim * 3)

        user_out1 = self.layer1_concatenator(user_concat_1)
        user_out1 = self.dropout(self.relu(self.bn1_user(user_out1)))
        # shape: (num_users, out1_dim)

        # --- 아이템 헤드 수행 ---
        item_h1_1 = h1_1['item'].squeeze(1)  # shape: (num_items, out1_dim)
        item_h1_2 = h1_2['item'].squeeze(1)
        item_h1_3 = h1_3['item'].squeeze(1)

        item_concat_1 = torch.cat([item_h1_1, item_h1_2, item_h1_3], dim=1)
        # shape: (num_items, out1_dim * 3)

        item_out1 = self.layer1_concatenator(item_concat_1)
        item_out1 = self.dropout(self.relu(self.bn1_item(item_out1)))
        # shape: (num_items, out1_dim)

        # 다음 레이어 입력용 dict
        h_layer1 = {
            self.type1_name: user_out1,  # (610, out1_dim)
            self.type2_name: item_out1   # (9742, out1_dim)
        }
        
        # ======================
        # 2) Layer 2 Forward
        # ======================
        h2_1 = self.layer2_head1(g, h_layer1)  # {'user':(610,1,out2_dim), 'item':(9742,1,out2_dim)}
        h2_2 = self.layer2_head2(g, h_layer1)
        h2_3 = self.layer2_head3(g, h_layer1)

        # --- 유저 헤드 수행 ---
        user_h2_1 = h2_1['user'].squeeze(1)  # (610, out2_dim)
        user_h2_2 = h2_2['user'].squeeze(1)
        user_h2_3 = h2_3['user'].squeeze(1)

        user_concat_2 = torch.cat([user_h2_1, user_h2_2, user_h2_3], dim=1)
        # shape: (610, out2_dim * 3)

        user_out2 = self.layer2_concatenator(user_concat_2)
        user_out2 = self.dropout(self.relu(self.bn2_user(user_out2)))
        # shape: (610, out2_dim)

        # --- 아이템 헤드 수행 ---
        item_h2_1 = h2_1['item'].squeeze(1)  # (9742, out2_dim)
        item_h2_2 = h2_2['item'].squeeze(1)
        item_h2_3 = h2_3['item'].squeeze(1)

        item_concat_2 = torch.cat([item_h2_1, item_h2_2, item_h2_3], dim=1)
        # shape: (9742, out2_dim * 3)

        item_out2 = self.layer2_concatenator(item_concat_2)
        item_out2 = self.dropout(self.relu(self.bn2_item(item_out2)))
        # shape: (9742, out2_dim)

        # ========== 최종 반환 ==========
        h_final = {
            self.type1_name: user_out2,  # (610, out2_dim)
            self.type2_name: item_out2   # (9742, out2_dim)
        }
        return h_final



from .RatingEncodingGAT import RatingEncodingGATConv

class DiffHeadGATRating(nn.Module):
    def __init__(self, 
                 num_user_nodes : int, 
                 num_location_nodes : int, 
                 emb_dim : int,       # 임베딩 차원
                 out1_dim : int,      # 첫 번째 레이어 출력 차원
                 out2_dim : int,      # 두 번째 레이어 출력 차원
                 uid2ts : dict, 
                 uid2dg : dict,
                 mid2dg : dict,
                 uid2rt : dict,
                 mid2rt : dict,
                 device : str,

                 allow_zero_in_degree = True,
                 feat_drop = 0.2,
                 attn_drop = 0.2,
                 negative_slope = 0.01,
                 residual = True,
                 dropout=0.2):
        
        super(DiffHeadGATRating, self).__init__()
        
        # ============ 유저별 정규화된 타임스탬프 딕셔너리 ============
        self.uid2ts = uid2ts
        self.uid2dg = uid2dg
        self.mid2dg = mid2dg

        self.uid2rt = uid2rt
        self.mid2rt = mid2rt

        # ============ 이종그래프 타입 정의 ============
        self.type1_name = "user"
        self.type2_name = "item"

        # ============ 각 타입별 특징 임베딩 ============ 
        self.emb_dim = emb_dim
        self.type1_emb = nn.Embedding(num_user_nodes, emb_dim)
        self.type2_emb = nn.Embedding(num_location_nodes, emb_dim)

        # ============ 각 레이어별 출력 차원 수 ============ 
        self.out1_dim = out1_dim
        self.out2_dim = out2_dim

        # ============ 각 타입 별 노드 수 ============ 
        self.num_user_nodes = num_user_nodes
        self.num_location_nodes = num_location_nodes

        # ============ 드롭아웃 레이어 ============ 
        self.dropout = nn.Dropout(dropout)

        # ============ 활성화함수 정의 ============ 
        self.relu = nn.SiLU()

        # ============ [ GAT Layer 1 ] ============
        # 첫 번째 HeteroGraphConv 레이어: emb_dim -> out_dim
        self.layer1_head1 = HeteroGraphConv({
            'go':   GATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual),

            'back': GATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual)
        }, aggregate='mean') 

        self.layer1_head2 = HeteroGraphConv({
            'go':   RatingEncodingGATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2rt = self.uid2rt, mid2rt = self.mid2rt, device = device),

            'back': RatingEncodingGATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2rt = self.uid2rt, mid2rt = self.mid2rt, device = device)
        }, aggregate='mean')  
 
        
        self.layer1_head3 = HeteroGraphConv({
            'go':   PopEncodingGATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2dg = self.uid2dg, mid2dg = self.mid2dg, device = device),

            'back': PopEncodingGATConv(in_feats=emb_dim, out_feats=out1_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2dg = self.uid2dg, mid2dg = self.mid2dg, device = device)
        }, aggregate='mean')  

        self.layer1_concatenator = nn.Linear(out1_dim * 3, out1_dim)

        # 첫 번째 레이어의 출력에 대한 BatchNorm
        self.bn1_user = nn.BatchNorm1d(self.out1_dim)
        self.bn1_item = nn.BatchNorm1d(self.out1_dim)

        # ============ [ Layer 2 ] ============
        self.layer2_head1 = HeteroGraphConv({
            'go':   GATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual),

            'back': GATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual)
        }, aggregate='mean')

        self.layer2_head2 = HeteroGraphConv({
            'go':   RatingEncodingGATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2rt = self.uid2rt, mid2rt = self.mid2rt, device = device),

            'back': RatingEncodingGATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2rt = self.uid2rt, mid2rt = self.mid2rt, device = device)
        }, aggregate='mean')

        self.layer2_head3 = HeteroGraphConv({
            'go':   PopEncodingGATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2dg = self.uid2dg, mid2dg = self.mid2dg, device = device),

            'back': PopEncodingGATConv(in_feats=out1_dim, out_feats=out2_dim, num_heads=1,
                            feat_drop=feat_drop, attn_drop=attn_drop,
                            negative_slope=negative_slope, residual=residual,
                            num_users = self.num_user_nodes, num_items = self.num_location_nodes,
                            uid2dg = self.uid2dg, mid2dg = self.mid2dg, device = device)
        }, aggregate='mean')

        self.layer2_concatenator = nn.Linear(out2_dim * 3, out2_dim)

        # 두 번째 레이어의 출력에 대한 BatchNorm
        self.bn2_user = nn.BatchNorm1d(self.out2_dim)
        self.bn2_item = nn.BatchNorm1d(self.out2_dim)


    def forward(self, g):

        inputs = {
            self.type1_name: self.type1_emb.weight,  
            self.type2_name: self.type2_emb.weight   
        }

        # ======================
        # 1) Layer 1 Forward
        # ======================

        h1_1 = self.layer1_head1(g, inputs)  # {'user':(610,1,out1_dim), 'item':(9742,1,out1_dim)}
        h1_2 = self.layer1_head2(g, inputs)
        h1_3 = self.layer1_head3(g, inputs)

        # --- 유저 헤드 수행 ---
        user_h1_1 = h1_1['user'].squeeze(1)  # shape: (num_users, out1_dim)
        user_h1_2 = h1_2['user'].squeeze(1)
        user_h1_3 = h1_3['user'].squeeze(1)

        user_concat_1 = torch.cat([user_h1_1, user_h1_2, user_h1_3], dim=1)
        # shape: (num_users, out1_dim * 3)

        user_out1 = self.layer1_concatenator(user_concat_1)
        user_out1 = self.dropout(self.relu(self.bn1_user(user_out1)))
        # shape: (num_users, out1_dim)

        # --- 아이템 헤드 수행 ---
        item_h1_1 = h1_1['item'].squeeze(1)  # shape: (num_items, out1_dim)
        item_h1_2 = h1_2['item'].squeeze(1)
        item_h1_3 = h1_3['item'].squeeze(1)

        item_concat_1 = torch.cat([item_h1_1, item_h1_2, item_h1_3], dim=1)
        # shape: (num_items, out1_dim * 3)

        item_out1 = self.layer1_concatenator(item_concat_1)
        item_out1 = self.dropout(self.relu(self.bn1_item(item_out1)))
        # shape: (num_items, out1_dim)

        # 다음 레이어 입력용 dict
        h_layer1 = {
            self.type1_name: user_out1,  # (610, out1_dim)
            self.type2_name: item_out1   # (9742, out1_dim)
        }

        # ======================
        # 2) Layer 2 Forward
        # ======================
        h2_1 = self.layer2_head1(g, h_layer1)  # {'user':(610,1,out2_dim), 'item':(9742,1,out2_dim)}
        h2_2 = self.layer2_head2(g, h_layer1)
        h2_3 = self.layer2_head3(g, h_layer1)

        # --- 유저 헤드 수행 ---
        user_h2_1 = h2_1['user'].squeeze(1)  # (610, out2_dim)
        user_h2_2 = h2_2['user'].squeeze(1)
        user_h2_3 = h2_3['user'].squeeze(1)
        
        user_concat_2 = torch.cat([user_h2_1, user_h2_2, user_h2_3], dim=1)
        # shape: (610, out2_dim * 3)

        user_out2 = self.layer2_concatenator(user_concat_2)
        user_out2 = self.dropout(self.relu(self.bn2_user(user_out2)))
        # shape: (610, out2_dim)

        # --- 아이템 헤드 수행 ---
        item_h2_1 = h2_1['item'].squeeze(1)  # (9742, out2_dim)
        item_h2_2 = h2_2['item'].squeeze(1)
        item_h2_3 = h2_3['item'].squeeze(1)

        item_concat_2 = torch.cat([item_h2_1, item_h2_2, item_h2_3], dim=1)
        # shape: (9742, out2_dim * 3)

        item_out2 = self.layer2_concatenator(item_concat_2)
        item_out2 = self.dropout(self.relu(self.bn2_item(item_out2)))
        # shape: (9742, out2_dim)

        # ========== 최종 반환 ==========
        h_final = {
            self.type1_name: user_out2,  # (610, out2_dim)
            self.type2_name: item_out2   # (9742, out2_dim)
        }
        return h_final