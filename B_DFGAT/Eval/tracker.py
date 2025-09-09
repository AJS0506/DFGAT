import matplotlib.pyplot as plt

class TrainTracker():
    def __init__(self,
                 train_loss_tracker, val_loss_tracker,
                 macro_recall, macro_precision, macro_ndcg,
                 micro_recall, micro_precision, micro_ndcg,
                 folder_path):
        """
        매개변수 설명:
            train_loss_tracker     : 각 epoch별 train_loss 리스트
            val_loss_tracker       : 각 epoch별 val_loss 리스트
            macro_recall, macro_precision, macro_ndcg
                                  : Top-K 리스트에 대응되는 Macro 지표 리스트
            micro_recall, micro_precision, micro_ndcg
                                  : Top-K 리스트에 대응되는 Micro 지표 리스트
            folder_path           : 이미지나 모델 결과를 저장할 폴더 경로
        """
        self.train_loss_tracker = train_loss_tracker
        self.val_loss_tracker = val_loss_tracker

        # Macro 지표
        self.macro_recall = macro_recall
        self.macro_precision = macro_precision
        self.macro_ndcg = macro_ndcg

        # Micro 지표
        self.micro_recall = micro_recall
        self.micro_precision = micro_precision
        self.micro_ndcg = micro_ndcg

        self.folder_path = folder_path

    def plot_results(self):
        """
        - num_epochs는 train_loss_tracker 길이로 계산
        - topk_list는 [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]로 고정
        - Loss 곡선, Macro 곡선, Micro 곡선을 각각 그려서 저장
        - 마지막에 tracker_value.txt에 모든 리스트 기록
        """

        num_epochs = len(self.train_loss_tracker)
        topk_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # ============ (1) 에포크별 Loss 그래프 ============ 
        plt.figure()
        plt.plot(range(1, num_epochs + 1), self.train_loss_tracker, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), self.val_loss_tracker, label='Val Loss')
        plt.title(f"Loss Curve (Up to Epoch {num_epochs})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.folder_path}/loss_curve.png")
        plt.close()

        # ============ (2) MACRO Top-K 그래프 ============ 
        plt.figure()
        plt.plot(topk_list, self.macro_recall, label='Macro Recall')
        plt.plot(topk_list, self.macro_precision, label='Macro Precision')
        plt.plot(topk_list, self.macro_ndcg, label='Macro NDCG')
        plt.title(f"Macro-Averaged Metrics (Up to Epoch {num_epochs})")
        plt.xlabel("Top-K")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.folder_path}/macro_metrics_curve.png")
        plt.close()

        # ============ (3) MICRO Top-K 그래프 ============ 
        plt.figure()
        plt.plot(topk_list, self.micro_recall, label='Micro Recall')
        plt.plot(topk_list, self.micro_precision, label='Micro Precision')
        plt.plot(topk_list, self.micro_ndcg, label='Micro NDCG')
        plt.title(f"Micro-Averaged Metrics (Up to Epoch {num_epochs})")
        plt.xlabel("Top-K")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.folder_path}/micro_metrics_curve.png")
        plt.close()

        # ============ (4) tracker_value.txt 파일에 리스트 기록 ============ 
        txt_file_path = f"{self.folder_path}/tracker_value.txt"
        with open(txt_file_path, "w") as f:
            f.write("===== Tracker Values =====\n\n")

            f.write("[Train Loss Tracker]\n")
            f.write(str(self.train_loss_tracker) + "\n\n")

            f.write("[Val Loss Tracker]\n")
            f.write(str(self.val_loss_tracker) + "\n\n")

            # ---- Macro 지표 ----
            f.write("[Macro Recall]\n")
            f.write(str(self.macro_recall) + "\n\n")

            f.write("[Macro Precision]\n")
            f.write(str(self.macro_precision) + "\n\n")

            f.write("[Macro NDCG]\n")
            f.write(str(self.macro_ndcg) + "\n\n")

            # ---- Micro 지표 ----
            f.write("[Micro Recall]\n")
            f.write(str(self.micro_recall) + "\n\n")

            f.write("[Micro Precision]\n")
            f.write(str(self.micro_precision) + "\n\n")

            f.write("[Micro NDCG]\n")
            f.write(str(self.micro_ndcg) + "\n\n")

        print(f"Plots saved and tracker values written in {txt_file_path}")


# import matplotlib.pyplot as plt

# class TrainTracker():
#     def __init__(self, train_loss_tracker, val_loss_tracker, avg_recall, avg_precision, avg_ndcg, folder_path):
#         """
#         매개변수 설명:
#             train_loss_tracker  : 각 epoch별 train_loss 리스트
#             val_loss_tracker    : 각 epoch별 val_loss 리스트
#             avg_recall          : topK 리스트에 대응되는 recall 리스트
#             avg_precision       : topK 리스트에 대응되는 precision 리스트
#             avg_ndcg            : topK 리스트에 대응되는 ndcg 리스트
#             folder_path         : 이미지나 모델 결과를 저장할 폴더 경로
#         """
#         self.train_loss_tracker = train_loss_tracker
#         self.val_loss_tracker = val_loss_tracker
#         self.avg_recall = avg_recall
#         self.avg_precision = avg_precision
#         self.avg_ndcg = avg_ndcg
#         self.folder_path = folder_path

#     def plot_results(self):
#         """
#         - num_epochs는 train_loss_tracker 길이로 계산
#         - topk_list는 [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]로 고정
#         - 두 가지 그래프(loss, metrics)를 각각 저장
#         - 마지막에 tracker_value.txt에 5개 리스트값 저장
#         """

#         num_epochs = len(self.train_loss_tracker)
#         topk_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#         # ============ (1) 에포크별 Loss 그래프 ============
#         plt.figure()
#         plt.plot(range(1, num_epochs + 1), self.train_loss_tracker, label='Train Loss')
#         plt.plot(range(1, num_epochs + 1), self.val_loss_tracker, label='Val Loss')
#         plt.title(f"Loss Curve (Up to Epoch {num_epochs})")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"{self.folder_path}/loss_curve.png")
#         plt.close()

#         # ============ (2) Top-K별 Recall, Precision, NDCG 그래프 ============
#         plt.figure()
#         plt.plot(topk_list, self.avg_recall, label='Recall')
#         plt.plot(topk_list, self.avg_precision, label='Precision')
#         plt.plot(topk_list, self.avg_ndcg, label='NDCG')
#         plt.title(f"Top-K Metrics (Up to Epoch {num_epochs})")
#         plt.xlabel("Top-K")
#         plt.ylabel("Metric Value")
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"{self.folder_path}/metrics_curve.png")
#         plt.close()

#         # ============ (3) tracker_value.txt 파일에 리스트 기록 ============
#         txt_file_path = f"{self.folder_path}/tracker_value.txt"
#         with open(txt_file_path, "w") as f:
#             f.write("===== Tracker Values =====\n\n")
#             f.write("[Train Loss Tracker]\n")
#             f.write(str(self.train_loss_tracker) + "\n\n")

#             f.write("[Val Loss Tracker]\n")
#             f.write(str(self.val_loss_tracker) + "\n\n")

#             f.write("[Avg Recall]\n")
#             f.write(str(self.avg_recall) + "\n\n")

#             f.write("[Avg Precision]\n")
#             f.write(str(self.avg_precision) + "\n\n")

#             f.write("[Avg NDCG]\n")
#             f.write(str(self.avg_ndcg) + "\n\n")

#         print(f"Tracker values saved to {txt_file_path}")
