import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, HeteroGraphConv

import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class TwoLayerSimpleLightGCN(nn.Module):
    """
    간단한 2-Layer LightGCN 구현 예시 (user, item 2개 노드타입)
    - 초기 임베딩: emb_dim
    - Layer1, Layer2 에서 Weight transform 없음, 이웃 임베딩 평균
    - 최종 임베딩: Layer0 + Layer1 + Layer2 합
    """
    def __init__(
        self,
        num_user_nodes,
        num_item_nodes,
        emb_dim,
        dataset="gowalla"
    ):
        super(TwoLayerSimpleLightGCN, self).__init__()
        
        # 노드 타입명 설정 (사용자 환경에 따라 수정)
        self.type1_name = "user"
        self.type2_name = "item"
        
        self.emb_dim = emb_dim
        
        # 유저 / 아이템 임베딩
        self.user_emb = nn.Embedding(num_user_nodes, emb_dim)
        self.item_emb = nn.Embedding(num_item_nodes, emb_dim)
        
        # LightGCN은 별도 Weight나 Bias, Activation이 없음
        # nn.Parameter(...)나 BatchNorm 등을 두지 않음.

        # 임베딩 초기화 (기본적으로 PyTorch는 uniform / normal 등으로 초기화)
        # 필요시 nn.init.xavier_uniform_(self.user_emb.weight), 등 직접 설정 가능
        
    def forward(self, g):
        """
        g: user-item 이분법 HeteroGraph
           - 에지 타입: (user, 'go', item), (item, 'back', user) 등
           - self-loop 등 없음
        
        Layer0 임베딩:  E^0_u, E^0_i  (초기값)
        Layer1 임베딩:  이웃의 Layer0 임베딩 평균
        Layer2 임베딩:  이웃의 Layer1 임베딩 평균
        
        최종 임베딩 = Layer0 + Layer1 + Layer2
        """
        device = g.device
        
        # --- Layer 0 (초기 임베딩) ---
        E0_user = self.user_emb.weight  # shape: [num_users, emb_dim]
        E0_item = self.item_emb.weight  # shape: [num_items, emb_dim]
        
        # HeteroGraph 상에서 user, item 임베딩 dict로 관리
        h_dict_0 = {
            self.user_type: E0_user,
            self.item_type: E0_item
        }
        
        # --- Layer 1 ---
        # user <- item, item <- user 이웃 임베딩을 평균
        h_dict_1 = self.propagate_one_layer(g, h_dict_0)
        # shape (user): [num_users, emb_dim], (item): [num_items, emb_dim]
        
        # --- Layer 2 ---
        # 방금 구한 Layer1 임베딩을 이웃에서 받아 평균
        h_dict_2 = self.propagate_one_layer(g, h_dict_1)
        
        # --- 최종 임베딩 ---
        # LightGCN은 layer0 + layer1 + layer2를 합쳐 사용 (또는 평균)
        final_user = (h_dict_0[self.user_type]
                      + h_dict_1[self.user_type]
                      + h_dict_2[self.user_type]) / 3.0   # 여기선 '평균'으로

        final_item = (h_dict_0[self.item_type]
                      + h_dict_1[self.item_type]
                      + h_dict_2[self.item_type]) / 3.0
        
        return {
            self.user_type: final_user,
            self.item_type: final_item
        }
    
    def propagate_one_layer(self, g, h_dict):
        """
        LightGCN 1 layer propagation:
        각 노드 타입별로 이웃 임베딩을 평균받음.
        """
        # h_dict: { "user": (U, d), "item": (I, d) }
        # 출력도 동일 형태의 dict
        with g.local_scope():
            # 노드별 임베딩 등록
            g.nodes[self.user_type].data['h'] = h_dict[self.user_type]
            g.nodes[self.item_type].data['h'] = h_dict[self.item_type]
            
            # Edge 방향에 맞춰 메시지 전달
            # user -> item ('go')
            # item -> user ('back')
            
            # user->item
            g.apply_edges(fn.u_copy_u('h', 'm'), etype=('user', 'go', self.item_type))
            # item->user
            g.apply_edges(fn.u_copy_u('h', 'm'), etype=(self.item_type, 'back', self.user_type))
            
            # 이후 edge->dst로 메시지 평균
            # user->item
            g.update_all(
                message_func=fn.copy_e('m', 'm'),
                reduce_func=fn.mean('m', 'h_new'),
                etype=('user', 'go', self.item_type)
            )
            # item->user
            g.update_all(
                message_func=fn.copy_e('m', 'm'),
                reduce_func=fn.mean('m', 'h_new'),
                etype=(self.item_type, 'back', self.user_type)
            )
            
            # 결과 추출
            h_user_out = g.nodes[self.user_type].data['h_new']
            h_item_out = g.nodes[self.item_type].data['h_new']
        
        return {
            self.user_type: h_user_out,
            self.item_type: h_item_out
        }
