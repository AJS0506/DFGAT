import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
import math

class PopEncodingGATConv(GATConv):
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 num_heads: int,
                 feat_drop: float,
                 attn_drop: float,
                 negative_slope: float,
                 residual: bool,
                 num_users: int,
                 num_items: int,
                 uid2dg: dict,
                 mid2dg: dict,
                 device: str,
                 activation=None,
                 allow_zero_in_degree: bool = False,
                 bias: bool = True):

        super().__init__(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
            bias=bias,
        )
        
        # 인코딩 더할 특징 차원 (TimeEncoding과 동일하게 입력 차원과 동일)
        self.encoding_dim = in_feats

        # 유저 수 및 아이템 수
        self.num_users = num_users
        self.num_items = num_items

        # 노드 차수 저장
        self.uid2dg = uid2dg
        self.mid2dg = mid2dg
        
        # 전역 통계량 계산 (정규화용)
        all_user_degrees = list(uid2dg.values())
        all_item_degrees = list(mid2dg.values())
        
        # 유저와 아이템의 전역 평균 및 표준편차 계산
        user_mean = sum(all_user_degrees) / len(all_user_degrees) if all_user_degrees else 0
        item_mean = sum(all_item_degrees) / len(all_item_degrees) if all_item_degrees else 0
        
        user_std = math.sqrt(sum((d - user_mean) ** 2 for d in all_user_degrees) / len(all_user_degrees)) if all_user_degrees else 1
        item_std = math.sqrt(sum((d - item_mean) ** 2 for d in all_item_degrees) / len(all_item_degrees)) if all_item_degrees else 1
        
        # 0으로 나누기 방지
        user_std = max(user_std, 1e-5)
        item_std = max(item_std, 1e-5)
        
        # 각 노드에 대해 저장하는 정보:
        # 1. 원래 데이터의 차수 (유저, 영화별로)
        # 2. 로그 변환 차수 -> 큰거 작게만들기 위해서
        # 3. Z-점수 정규화 차수 -> train / val / test를 0~1로하면 문제가 생겨서 Z스코어로 정규화함
        # 4. 상대적 인기도 -> 전체 상호작용중에 내 상호작용이 얼마인가?
        # 각 유저별로 [유저, 4] 행렬을 0으로 채움

        user_stats = torch.zeros(num_users, 4)
        item_stats = torch.zeros(num_items, 4)
        
        total_user_interactions = sum(all_user_degrees)
        total_item_interactions = sum(all_item_degrees)
        
        # 유저 정보 4개 추가!
        # 각 행에 4개만들어뒀던 값에 대입
        for uid, degree in uid2dg.items():
            user_stats[uid, 0] = degree
            user_stats[uid, 1] = math.log(degree + 1)  # log(0) 방지
            user_stats[uid, 2] = (degree - user_mean) / user_std  # Z-점수
            user_stats[uid, 3] = degree / total_user_interactions if total_user_interactions > 0 else 0  # 상대적 인기도
        
        # 아이템 정보 4개 추가!
        for mid, degree in mid2dg.items():
            item_stats[mid, 0] = degree
            item_stats[mid, 1] = math.log(degree + 1)  # log(0) 방지
            item_stats[mid, 2] = (degree - item_mean) / item_std  # Z-점수
            item_stats[mid, 3] = degree / total_item_interactions if total_item_interactions > 0 else 0  # 상대적 인기도
        
        # GPU 이동을 위한 버퍼 등록
        self.register_buffer('user_stats', user_stats)
        self.register_buffer('item_stats', item_stats)
        
        # 유저 및 아이템 통계에 대한 선형 변환
        # 유저별 4개의 특징 -> 유저 히든 피쳐의 개수로 확장하는 선형 레이어
        self.user_linear = nn.Linear(4, self.encoding_dim)
        self.item_linear = nn.Linear(4, self.encoding_dim)
         
    def forward(self, graph, feat, *args, **kwargs):
        # tuple → 개별 텐서
        h_type1, h_type2 = feat
        
        # 인기도 임베딩 얻기
        user_emb = self.user_linear(self.user_stats)
        item_emb = self.item_linear(self.item_stats)
        
        # h_type1이 유저에 해당하는지 확인 (TimeEncodingGATConv와 유사)
        if h_type1.shape[0] == self.num_users:
            h_type1 = h_type1 + user_emb
            h_type2 = h_type2 + item_emb
        else:
            h_type1 = h_type1 + item_emb
            h_type2 = h_type2 + user_emb
         
        # 업데이트된 특성으로 전달
        return super().forward(graph, (h_type1, h_type2), *args, **kwargs)