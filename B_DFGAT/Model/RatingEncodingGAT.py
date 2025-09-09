import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv

class RatingEncodingGATConv(GATConv):
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
                 uid2rt: dict,  # 평점 정보 딕셔너리 
                 mid2rt: dict,  # 영화 평점 정보 딕셔너리
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

        # 인코딩 차원은, Feature에 더해줘야 하므로 맞춥니다
        self.encoding_dim = in_feats

        # 유저 수 및 영화 수
        self.num_users = num_users
        self.num_items = num_items

        self.uid2rt = uid2rt
        self.user_rt_mean = {}
        self.user_rt_var = {}

        # 1) 유저에 대한 평점 평균과 분산 구하기
        for uid, rt_list in uid2rt.items():
            rt = torch.tensor(rt_list, dtype=torch.float32)
            self.user_rt_mean[uid] = rt.mean().item()
            self.user_rt_var[uid] = rt.var(unbiased=False).item()
                                                                  
        # 2) [num_users, 2] 텐서로 묶어서 buffer 로 저장
        stats = torch.zeros(num_users, 2)
        for uid in range(num_users):
            stats[uid, 0] = self.user_rt_mean.get(uid, 0.0)
            stats[uid, 1] = self.user_rt_var.get(uid, 0.0)

        self.register_buffer('user_rt_stats', stats)
        self.rt_linear = nn.Linear(2, self.encoding_dim)
         
    def forward(self, graph, feat, *args, **kwargs):
        # tuple → 개별 텐서
        h_type1, h_type2 = feat
        
        rt_emb = self.rt_linear(self.user_rt_stats)

        if h_type1.shape[0] == self.num_users:
            h_type1 = h_type1 + rt_emb
        else:
            h_type2 = h_type2 + rt_emb
         
        # 새 tuple 로 묶어 넘기기
        return super().forward(graph, (h_type1, h_type2), *args, **kwargs)