# eval_calc.py
import torch, math, os
from collections import defaultdict

class calcEvaluationScore():
    """
    평가 클래스(수정본)
    - relevant(정답) 정의: 평점 >= pos_threshold (기본 4.0). implicit 데이터(gowalla)는 전부 정답.
    - 추천 후보에서 train(+val)에서 이미 본 아이템은 제외(masking).
    - Macro/Micro: Recall@K, Precision@K, NDCG@K 모두 계산.
    """
    def __init__(
        self,
        dataset,                      # 테스트셋 raw tuples
        hidden_state,                 # {'user': UxD, 'item': IxD}
        folder_path,                  # 결과 저장 폴더
        dataset_name,                 # 데이터셋 이름(예: "filmtrust", "gowalla")
        seen_train=None,              # {u: set(items)} - train에서 본 아이템
        seen_val=None,                # {u: set(items)} - val에서 본 아이템
        pos_threshold: float = 4.0,   # 평점 임계값(implicit이면 무시)
        exclude_seen_in_reco: bool = True,  # 후보 마스킹 여부
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.user_h = hidden_state['user']   # [num_users, dim]
        self.movie_h = hidden_state['item']  # [num_items, dim]

        self.pos_threshold = pos_threshold
        self.exclude_seen_in_reco = exclude_seen_in_reco

        # 본 아이템(후보에서 제외할 집합)
        self.seen_train = seen_train or defaultdict(set)
        self.seen_val   = seen_val   or defaultdict(set)

        # 평가할 Top-K 리스트
        self.topk = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # (기존) Macro-averaging 결과
        self.avg_recall = []
        self.avg_precision = []
        self.avg_ndcg = []

        # (추가) Micro-averaging 결과
        self.micro_avg_recall = []
        self.micro_avg_precision = []
        self.micro_avg_ndcg = []

        # 결과 저장 경로
        self.txt_file_path = os.path.join(folder_path, "eval.txt")
        with open(self.txt_file_path, 'w', encoding='utf-8') as f:
            f.write("==== Model Evaluation Scores ===\n\n\n")

    def _build_user_truth(self):
        """
        사용자별 정답(테스트) 아이템 목록을 구성.
        - explicit: 평점 >= pos_threshold만 정답으로 사용
        - implicit(gowalla 등): 모두 정답으로 사용
        - 중복 제거
        """
        test_set = defaultdict(list)

        if self.dataset_name == "gowalla":  # implicit
            for uid, mid, _ in self.dataset:
                test_set[uid].append(mid)
        else:  # explicit
            th = float(self.pos_threshold)
            for uid, mid, _, rt in self.dataset:
                if float(rt) >= th:
                    test_set[uid].append(mid)

        # 중복 제거 및 빈 유저 제거
        cleaned = {}
        for u, items in test_set.items():
            uniq = list(set(items))
            if len(uniq) > 0:
                cleaned[u] = uniq
        return cleaned

    @staticmethod
    def _idcg_at_k(k: int) -> float:
        """정규화 상수 IDCG@K (binary relevance)"""
        idcg = 0.0
        for rank in range(1, k + 1):
            idcg += 1.0 / math.log2(rank + 1)
        return idcg

    def calcScore(self):
        # --------------------------------------
        # 1) 유저별 테스트 정답 목록 만들기 (임계값 반영)
        # --------------------------------------
        user_truth = self._build_user_truth()
        num_items = self.movie_h.shape[0]

        # 미리 user-level IDCG 캐시(속도)
        idcg_cache = {}

        # --------------------------------------
        # 2) Top-K별로 반복
        # --------------------------------------
        for topk in self.topk:
            # 매크로/마이크로 누적 변수
            macro_recall_sum = 0.0
            macro_precision_sum = 0.0
            macro_ndcg_sum = 0.0
            macro_user_count = 0  # 정답 없는 유저는 스킵

            total_intersect = 0   # Micro: 전체 맞춘 수
            total_truth = 0       # Micro: 전체 정답 수
            total_predict = 0     # Micro: 전체 예측 수(= sum of allowed_k per user)
            sum_DCG = 0.0
            sum_IDCG = 0.0

            # --------------------------------------
            # 2-1) 모든 유저에 대해 반복
            # --------------------------------------
            for user, truth_items in user_truth.items():
                truth_set = set(truth_items)
                num_truth = len(truth_set)
                if num_truth == 0:
                    continue  # 안전

                # 유저 점수 계산 (u·v)
                user_vec = self.user_h[user]              # [D]
                scores = (user_vec * self.movie_h).sum(dim=1)  # [I]

                # 후보에서 train(+val)에서 본 아이템 제외
                allowed_k = topk
                if self.exclude_seen_in_reco:
                    seen = (self.seen_train.get(user, set())
                            | self.seen_val.get(user, set()))
                    if seen:
                        # 범위 내 인덱스만 취함(안전)
                        seen_idx = [i for i in seen if 0 <= i < num_items]
                        if len(seen_idx) > 0:
                            idx = torch.tensor(seen_idx, dtype=torch.long, device=scores.device)
                            scores.index_fill_(0, idx, float('-inf'))
                        # 남은 후보 수 계산(음수 방지)
                        candidate_cnt = max(0, num_items - len(seen_idx))
                        allowed_k = min(topk, max(1, candidate_cnt))
                    else:
                        allowed_k = min(topk, num_items)
                else:
                    allowed_k = min(topk, num_items)

                # 상위 K 추출
                _, topk_indices = torch.topk(scores, allowed_k, largest=True)
                topk_list = topk_indices.tolist()

                # 교집합
                intersect_count = len(truth_set.intersection(topk_list))

                # ===== Macro Recall/Precision =====
                recall_k = intersect_count / num_truth
                precision_k = intersect_count / allowed_k

                # ===== NDCG@K (binary relevance) =====
                DCG = 0.0
                for rank, item_idx in enumerate(topk_list, start=1):
                    if item_idx in truth_set:
                        DCG += 1.0 / math.log2(rank + 1)

                max_rel = min(num_truth, allowed_k)
                if max_rel not in idcg_cache:
                    idcg_cache[max_rel] = self._idcg_at_k(max_rel)
                IDCG = idcg_cache[max_rel] if max_rel > 0 else 0.0
                ndcg_k = (DCG / IDCG) if IDCG > 0 else 0.0

                # ===== Macro 누적 =====
                macro_recall_sum += recall_k
                macro_precision_sum += precision_k
                macro_ndcg_sum += ndcg_k
                macro_user_count += 1

                # ===== Micro 누적 =====
                total_intersect += intersect_count
                total_truth += num_truth
                total_predict += allowed_k
                sum_DCG += DCG
                sum_IDCG += IDCG

            # --------------------------------------
            # 2-2) Macro 결과 계산(정답 있는 유저만 평균)
            # --------------------------------------
            if macro_user_count > 0:
                macro_recall = macro_recall_sum / macro_user_count
                macro_precision = macro_precision_sum / macro_user_count
                macro_ndcg = macro_ndcg_sum / macro_user_count
            else:
                macro_recall = macro_precision = macro_ndcg = 0.0

            # --------------------------------------
            # 2-3) Micro 결과 계산
            # --------------------------------------
            micro_recall = (total_intersect / total_truth) if total_truth > 0 else 0.0
            micro_precision = (total_intersect / total_predict) if total_predict > 0 else 0.0
            micro_ndcg = (sum_DCG / sum_IDCG) if sum_IDCG > 0 else 0.0

            # --------------------------------------
            # 2-4) 결과 저장
            # --------------------------------------
            self.avg_recall.append(macro_recall)
            self.avg_precision.append(macro_precision)
            self.avg_ndcg.append(macro_ndcg)

            self.micro_avg_recall.append(micro_recall)
            self.micro_avg_precision.append(micro_precision)
            self.micro_avg_ndcg.append(micro_ndcg)

            # --------------------------------------
            # 2-5) 콘솔 및 파일 출력
            # --------------------------------------
            with open(self.txt_file_path, 'a', encoding='utf-8') as fp:
                print("=========================================")
                print(f"[Top-K = {topk}]")
                print("               Macro    vs     Micro     \n")
                print("         --------------------------------")
                print(f"Recall:    {macro_recall:8.4f}       {micro_recall:8.4f}")
                print(f"Precision: {macro_precision:8.4f}       {micro_precision:8.4f}")
                print(f"NDCG:      {macro_ndcg:8.4f}       {micro_ndcg:8.4f}")
                print("=========================================")

                fp.write("=========================================\n")
                fp.write(f"[Top-K = {topk}]\n")
                fp.write("               Macro    vs     Micro     \n")
                fp.write("         --------------------------------\n")
                fp.write(f"Recall:    {macro_recall:8.4f}       {micro_recall:8.4f}\n")
                fp.write(f"Precision: {macro_precision:8.4f}       {micro_precision:8.4f}\n")
                fp.write(f"NDCG:      {macro_ndcg:8.4f}       {micro_ndcg:8.4f}\n")
                fp.write("=========================================\n\n")

        # 반환: (매크로, 마이크로) 모두 반환
        return (self.avg_recall, self.avg_precision, self.avg_ndcg,
                self.micro_avg_recall, self.micro_avg_precision, self.micro_avg_ndcg)



# import torch, math, os
# from collections import defaultdict

# class calcEvaluationScore():
#     def __init__(self, testset, hidden_state, folder_path, dataset_name):
#         self.dataset = testset
#         self.dataset_name = dataset_name

#         self.user_h = hidden_state['user']
#         self.movie_h = hidden_state['item']

#         self.topk = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#         self.avg_recall = []
#         self.avg_precision = []
#         self.avg_ndcg = []
        
#         self.txt_file_path = os.path.join(folder_path, "eval.txt")

#         with open(self.txt_file_path, 'w', encoding='utf-8') as f:
#             f.write("==== Movel Evaluation Scores ===\n\n\n")  

#     def calcScore(self):
#         # 모든 유저별 테스트 셋
#         test_set = defaultdict(list)

#         if self.dataset_name == "gowalla":
#             for uid, mid, _ in self.dataset:
#                 test_set[uid].append(mid)
#         else:
#             for uid, mid, _, _ in self.dataset:
#                 test_set[uid].append(mid)

#         for topk in self.topk:
#             cur_avg_recall = 0
#             cur_avg_precision = 0
#             cur_avg_ndcg = 0

#             for user in test_set.keys():
#                 test_mids = torch.LongTensor(test_set[user])

#                 user_score = (self.user_h[user] * self.movie_h).sum(dim=1)
#                 topk_values, topk_indices = torch.topk(user_score, topk, largest=True)


#                 # ===== Recall@ 계산! =====
#                 intersect = set(test_mids.tolist()) & set(topk_indices.tolist())
#                 recall_k = len(intersect) / len(test_mids) 

#                 # ===== Precision@ 계산! =====
#                 precision_k = len(intersect) / topk

#                 # ===== NDCG@ 계산! =====
#                 DCG = 0.0 # -> 랭킹 순서별로 log 스케일을 반영한 점수부여! 랭킹이 낮은데 맞췄으면 감소된 로그 스케일로 점수 합산함
#                 for rank, item_idx in enumerate(topk_indices.tolist(), start=1):
#                     if item_idx in test_mids:
#                         DCG += 1.0 / math.log2(rank + 1)
                
#                 max_rel = min(len(test_mids), topk)

#                 # 최대의 DCG 값으로 나누어서 정규화를 수행
#                 IDCG = 0.0
#                 for rank in range(1, max_rel + 1):
#                     IDCG += 1.0 / math.log2(rank + 1)

#                 if IDCG > 0:
#                     ndcg_k = DCG / IDCG
#                 else:
#                     ndcg_k = 0.0

#                 cur_avg_recall += recall_k
#                 cur_avg_precision += precision_k
#                 cur_avg_ndcg += ndcg_k

#             cur_avg_recall = cur_avg_recall / len(test_set)
#             cur_avg_precision = cur_avg_precision / len(test_set)
#             cur_avg_ndcg = cur_avg_ndcg / len(test_set)

#             with open(self.txt_file_path, 'a', encoding='utf-8') as fp:
#                 # 콘솔 출력
#                 print("=========================================")
#                 print(" Top-K |   Recall   | Precision |  NDCG ")
#                 print("=========================================")
#                 print(f"   {topk:3d} | {cur_avg_recall:10.4f} | {cur_avg_precision:9.4f} | {cur_avg_ndcg:6.4f}")

#                 # 파일 기록
#                 fp.write("=========================================\n")
#                 fp.write(" Top-K |   Recall   | Precision |  NDCG \n")
#                 fp.write("=========================================\n")
#                 fp.write(f"   {topk:3d} | {cur_avg_recall:10.4f} | {cur_avg_precision:9.4f} | {cur_avg_ndcg:6.4f}\n")
#                 fp.write("\n") 


#             self.avg_recall.append(cur_avg_recall)
#             self.avg_precision.append(cur_avg_precision)
#             self.avg_ndcg.append(cur_avg_ndcg)


#         return (self.avg_recall, self.avg_precision, self.avg_ndcg)

