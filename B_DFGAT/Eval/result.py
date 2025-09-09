import torch, math, os
from collections import defaultdict

class calcEvaluationScore():
    def __init__(self, dataset, hidden_state, folder_path, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.user_h = hidden_state['user']
        self.movie_h = hidden_state['item']

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

    def calcScore(self):
        # --------------------------------------
        # 1) 유저별 테스트 아이템 목록 만들기
        # --------------------------------------
        test_set = defaultdict(list)

        if self.dataset_name == "gowalla":
            for uid, mid, _ in self.dataset:
                test_set[uid].append(mid)
        else:
            for uid, mid, _, _ in self.dataset:
                test_set[uid].append(mid)

        # --------------------------------------
        # 2) Top-K별로 반복하며 평가
        # --------------------------------------
        for topk in self.topk:
            # (기존) Macro 평균을 위한 누적
            macro_recall_sum = 0.0
            macro_precision_sum = 0.0
            macro_ndcg_sum = 0.0

            # (추가) Micro-averaging을 위한 누적
            total_intersect = 0  # 전체 유저가 맞춘 아이템(교집합) 수의 합
            total_truth = 0      # 전체 유저 테스트 아이템(정답) 수의 합
            total_predict = 0    # 전체 유저가 예측한 아이템(Top-K) 수의 합
            sum_DCG = 0.0
            sum_IDCG = 0.0

            # --------------------------------------
            # 2-1) 모든 유저에 대해 반복 (UID 루프)
            # --------------------------------------
            for user in test_set.keys():
                test_mids = torch.LongTensor(test_set[user])

                # 유저 임베딩과 전체 아이템 임베딩 내적 → 상위 K개 인덱스
                user_score = (self.user_h[user] * self.movie_h).sum(dim=1)
                topk_values, topk_indices = torch.topk(user_score, topk, largest=True)

                # (공통) 교집합
                intersect = set(test_mids.tolist()) & set(topk_indices.tolist())
                num_hit = len(intersect)        # 이 유저가 맞춘 아이템 수
                num_truth = len(test_mids)      # 이 유저의 테스트 아이템 전체 개수
                # 위에서 topk가 현재 K값 → 즉 이 유저가 예측한 아이템 개수 = topk

                # ===== (A) Macro Recall@K, Precision@K =====
                recall_k = num_hit / num_truth
                precision_k = num_hit / topk

                # ===== (B) NDCG@K =====
                DCG = 0.0
                for rank, item_idx in enumerate(topk_indices.tolist(), start=1):
                    if item_idx in test_mids:
                        DCG += 1.0 / math.log2(rank + 1)
                max_rel = min(num_truth, topk)
                IDCG = 0.0
                for rank in range(1, max_rel + 1):
                    IDCG += 1.0 / math.log2(rank + 1)
                ndcg_k = (DCG / IDCG) if IDCG > 0 else 0.0

                # ===== (C) Macro 누적 =====
                macro_recall_sum += recall_k
                macro_precision_sum += precision_k
                macro_ndcg_sum += ndcg_k

                # ===== (D) Micro 누적 =====
                # - Recall: 전체 교집합 / 전체 정답
                # - Precision: 전체 교집합 / 전체 예측
                total_intersect += num_hit
                total_truth += num_truth
                total_predict += topk  # 모든 유저의 예측 아이템 수(=K)를 합산

                sum_DCG += DCG
                sum_IDCG += IDCG

            # --------------------------------------
            # 2-2) Macro 결과(기존 평균) 계산
            # --------------------------------------
            num_users = len(test_set)
            macro_recall = macro_recall_sum / num_users
            macro_precision = macro_precision_sum / num_users
            macro_ndcg = macro_ndcg_sum / num_users

            # --------------------------------------
            # 2-3) Micro 결과(전체 관점) 계산
            # --------------------------------------
            if total_truth > 0:
                micro_recall = total_intersect / total_truth
            else:
                micro_recall = 0.0

            if total_predict > 0:
                micro_precision = total_intersect / total_predict
            else:
                micro_precision = 0.0

            if sum_IDCG > 0:
                micro_ndcg = sum_DCG / sum_IDCG
            else:
                micro_ndcg = 0.0

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


        # 3) 반환: (매크로, 마이크로) 모두 반환!
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

