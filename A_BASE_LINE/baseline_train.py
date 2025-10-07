import argparse
import torch
import random
import copy
import os
import datetime
import json
from collections import defaultdict

# ============= 모델 임포트 =============
from Model.GCN import TwoLayerSimpleHeteroGCN
from Model.GAT import TwoLayerSimpleHeteroGAT
from Model.GraphSAGE import TwoLayerSimpleHeteroSAGE
from Model.GINC import TwoLayerSimpleHeteroGINC
from Model.Cheb import TwoLayerSimpleHeteroAGNNConv

from GraphMaker.make_graph import GraphMaker
from DataLoader.load_dataset import DataLoader

from Eval.result import calcEvaluationScore
from Eval.tracker import TrainTracker

# =================== argparse 설정 =============================
parser = argparse.ArgumentParser(description="Optional arguments for training.")
parser.add_argument("--gpu", type=int, default=-1, help="GPU ID (e.g., 0, 1, 2, 3) or -1 for CPU")
parser.add_argument("--dataset", type=int, default=1, help="Dataset index (0: gowalla, 1: movielens_small, 2: movielens_25M, 3: netflixPrize, 4: filmtrust)")
parser.add_argument("--seed", type=int, default=1004, help="Random seed")
parser.add_argument("--model", type=str, default="GCN", help="Model name (GCN/GAT/SAGE/GINC)")
parser.add_argument("--emb-dim", type=int, default=128, help="Embedding dimension")
parser.add_argument("--first-dim", type=int, default=64, help="First layer dimension")
parser.add_argument("--second-dim", type=int, default=32, help="Second layer dimension")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--early-stop", type=int, default=10, help="Early stopping patience")
parser.add_argument("--epochs", type=int, default=200, help="Maximum number of epochs")
args = parser.parse_args()

# 변수 할당
GPU_ID = args.gpu
DATASET_NUMBER = args.dataset
RANDOM_SEED = args.seed
SELECTED_MODEL = args.model
embedding_dim = args.emb_dim
first_layer_dim = args.first_dim
second_layer_dim = args.second_dim
LEARNING_RATE = args.lr
STOP_CONDITION = args.early_stop
epoch = args.epochs

# ============= 시드 고정 =============
def set_random_seed(seed):
    random.seed(seed)                # Python random 시드
    # np.random.seed(seed)             # NumPy 시드
    torch.manual_seed(seed)          # PyTorch CPU 시드
    torch.cuda.manual_seed(seed)     # PyTorch GPU 시드 (Single GPU)
    torch.cuda.manual_seed_all(seed) # Multi-GPU도 쓰는 경우
    torch.backends.cudnn.deterministic = True   # 연산 재현성 보장
    torch.backends.cudnn.benchmark = False      # 입력 크기가 일정할 때 최적화 비활성화

set_random_seed(RANDOM_SEED)

# ============= 데이터셋 정의 =============
dataset_list = ["gowalla", "movielens_small", "movielens_25M", "netflixPrize", "filmtrust"]
DATA_SET = dataset_list[DATASET_NUMBER]

# ============= 데이터셋으로 그래프 만들기 =============
gm = GraphMaker()
graph = gm.get_graph(dataset=DATA_SET)
num_type1_nodes, num_type2_nodes = gm.get_num_nodes()

print("학습 그래프 -> ",graph)

# ============= 학습 모델 초기화 및 GPU 설정 =============
if GPU_ID == -1:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
graph = graph.to(device)

models = {
    "GCN" : TwoLayerSimpleHeteroGCN(num_type1_nodes, num_type2_nodes, embedding_dim, first_layer_dim, second_layer_dim, DATA_SET).to(device),
    "GAT" : TwoLayerSimpleHeteroGAT(num_type1_nodes, num_type2_nodes, embedding_dim, first_layer_dim, second_layer_dim, DATA_SET).to(device),
    "SAGE" : TwoLayerSimpleHeteroSAGE(num_type1_nodes, num_type2_nodes, embedding_dim, first_layer_dim, second_layer_dim, DATA_SET).to(device),
    "GINC" : TwoLayerSimpleHeteroGINC(num_type1_nodes, num_type2_nodes, embedding_dim, first_layer_dim, second_layer_dim, DATA_SET).to(device),
    "AGNN" : TwoLayerSimpleHeteroAGNNConv(num_type1_nodes, num_type2_nodes, embedding_dim, first_layer_dim, second_layer_dim, DATA_SET).to(device),
}

if SELECTED_MODEL not in models:
    print(f"[Error] Model '{SELECTED_MODEL}' not found in models dict. model = [GCN, GAT, SAGE, GINC]")
    exit()

model = models[SELECTED_MODEL]

if GPU_ID == -1:
    print(f">>> Using CPU, Dataset: {DATASET_NUMBER}, Seed: {RANDOM_SEED}, Model: {SELECTED_MODEL}")
else:
    print(f">>> Using GPU: {GPU_ID}, Dataset: {DATASET_NUMBER}, Seed: {RANDOM_SEED}, Model: {SELECTED_MODEL}")

# ============= 옵티마이저 설정 =============
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# ============= 학습, 검증, 테스트 데이터셋 로드 =============
loader = DataLoader(DATA_SET)
train_set, val_set, test_set = loader.load_dataset()

# ============= Hard Negative 샘플링을 위한 데이터셋 딕셔너리 =============
user_visited = defaultdict(set)
for row in train_set + val_set + test_set:
    uid, mid = row[0], row[1]  # rating 있든 없든 처리 가능
    user_visited[uid].add(mid)

# ============= 에포크 및 배치사이즈 설정 =============
batch_size = len(train_set) // 30
print(f"훈련 세트 개수 -> {len(train_set)}, batch_size -> {batch_size}")

# ============= Loss Tracker =============
train_loss_tracker = []
val_loss_tracker = []

# ============= Early Stopping 구현 =============
early_stop_cnt = 0
before_val_loss = 0
STOP_CONDITION = 5 # 얼마나 참을 것 인지!

best_val_loss = float('inf')
best_model_state = None


# ============= 학습 시작! =============
for e in range(epoch):
    # ============================
    # 1) Training
    # ============================
    model.train()
    random.shuffle(train_set)  # 매 에폭마다 학습 데이터 순서를 무작위로 섞기
    num_samples = len(train_set)
    total_train_loss = 0.0

    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_data = train_set[i:end_idx]

        pos_users = torch.LongTensor([row[0] for row in batch_data]).to(device)
        pos_items = torch.LongTensor([row[1] for row in batch_data]).to(device)

        # GCN 호출
        h = model(graph)
        user_emb = h['user']
        item_emb = h['item']

        # =========== Random Hard Negative Sampling ===========
        neg_items_list = []
        pos_users_cpu = pos_users.cpu().tolist()  # 유저 ID를 CPU 리스트로 변환 (set/dict 접근 속도↑)

        for u in pos_users_cpu:
            visited_set = user_visited[u] # 이 유저가 이미 본 아이템 집합
            while True:
                candidate = random.randint(0, num_type2_nodes - 1)
                if candidate not in visited_set:
                    neg_items_list.append(candidate)
                    break

        neg_items = torch.tensor(neg_items_list, dtype=torch.long, device=device)

        # BPR loss 계산
        pos_scores = (user_emb[pos_users] * item_emb[pos_items]).sum(dim=1)
        neg_scores = (user_emb[pos_users] * item_emb[neg_items]).sum(dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        optimizer.zero_grad()
        bpr_loss.backward()
        optimizer.step()

        total_train_loss += bpr_loss.item()

    avg_train_loss = total_train_loss / (num_samples // batch_size + 1)

    train_loss_tracker.append(avg_train_loss)

    # ============================
    # 2) Validation
    # ============================
    model.eval()
    total_val_loss = 0.0
    val_samples = len(val_set)

    with torch.no_grad():
        for i in range(0, val_samples, batch_size):
            end_idx = min(i + batch_size, val_samples)
            val_batch = val_set[i:end_idx]

            pos_users = torch.LongTensor([row[0] for row in val_batch]).to(device)
            pos_items = torch.LongTensor([row[1] for row in val_batch]).to(device)

            # 모델 순전파
            h = model(graph)
            user_emb = h['user']
            item_emb = h['item']

            # 학습시와 동일한 하드 네거티브 샘플링!
            neg_items_list = []
            pos_users_cpu = pos_users.cpu().tolist()

            for u in pos_users_cpu:
                visited_set = user_visited[u]
                while True:
                    candidate = random.randint(0, num_type2_nodes - 1)
                    if candidate not in visited_set:
                        neg_items_list.append(candidate)
                        break

            neg_items = torch.tensor(neg_items_list, dtype=torch.long, device=device)

            pos_scores = (user_emb[pos_users] * item_emb[pos_items]).sum(dim=1)
            neg_scores = (user_emb[pos_users] * item_emb[neg_items]).sum(dim=1)
            val_bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            total_val_loss += val_bpr_loss.item()

    avg_val_loss = total_val_loss / (val_samples // batch_size + 1)

    val_loss_tracker.append(avg_val_loss)

    print(f"[Epoch {e+1}/inf] "
          f"Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {avg_val_loss:.4f}")

    # ============ Early Stopping 관련 코드 ================
    
    if e < 10:
        print("초기 학습.. (early stopping skip!)")
        continue

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"  >> Best model updated at epoch {e+1}, Val_loss = {avg_val_loss:.4f}")
        early_stop_cnt = 0  # 개선되었으므로 카운트 리셋
    else:
        early_stop_cnt += 1
        print("early stop cnt ->", early_stop_cnt)

    if early_stop_cnt >= STOP_CONDITION:
        print(f"early stop cnt reached {STOP_CONDITION}, stop")
        break




# ============ 가장 좋은 성능 불러오기 및 테스트 시작 ================

# Best 모델 state를 파일로 저장
model_class_name = model.__class__.__name__
time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 새로운 폴더 구조: BestModel/SEED/DATASET/MODEL/
folder_path = f"BestModel/{RANDOM_SEED}/{DATA_SET}/{SELECTED_MODEL}_{time_stamp}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 설정 정보를 JSON으로 저장
config = {
    "seed": RANDOM_SEED,
    "dataset": DATA_SET,
    "model": model_class_name,
    "learning_rate": LEARNING_RATE,
    "batch_size": batch_size,
    "stop_condition": STOP_CONDITION,
    "embedding_dim": embedding_dim,
    "first_layer_dim": first_layer_dim,
    "second_layer_dim": second_layer_dim,
    "timestamp": time_stamp,
    "gpu_id": GPU_ID,
    "selected_model": SELECTED_MODEL
}

# config.json 저장
with open(f"{folder_path}/config.json", 'w') as f:
    json.dump(config, f, indent=4)

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Best model parameters have been loaded.")

    # 모델명에 타임스탬프 추가하여 저장
    torch.save(best_model_state, f"{folder_path}/{model_class_name}_{time_stamp}.pth")
    print(f"최적 모델 저장 완료! -> {folder_path}/{model_class_name}_{time_stamp}.pth")


model.eval()
with torch.no_grad():
    test_hidden = model(graph)

# 1) 평가 ㄱㄱ
test_processor = calcEvaluationScore(test_set, test_hidden, folder_path)
(macro_recall, macro_precision, macro_ndcg,
 micro_recall, micro_precision, micro_ndcg) = test_processor.calcScore()

# 2) 트래커 ㄱㄱㄱ
tracker = TrainTracker(
    train_loss_tracker, 
    val_loss_tracker,
    macro_recall, macro_precision, macro_ndcg,
    micro_recall, micro_precision, micro_ndcg,
    folder_path
)

# 3) 결과 플롯 및 저장
tracker.plot_results()





# 랜덤 negative 샘플링!
# neg_items = torch.randint(0, num_type2_nodes, (len(val_batch),), device=device)
# mask = (neg_items == pos_items)
# while mask.any():
#     neg_items[mask] = torch.randint(0, num_type2_nodes, (mask.sum(),), device=device)
#     mask = (neg_items == pos_items)







# # 모든 아이템 집합
# all_items = set(range(num_type2_nodes))

# # 각 유저별로 아직 보지 않은 아이템 목록을 미리 만들어 둠
# unvisited_items = {}
# for u, visited_set in user_visited.items():
#     # user_visited[u] = 이미 본 아이템(set)
#     # unvisited_items[u] = 아직 안 본 아이템(list)

#     unvisited_items[u] = list(all_items - visited_set)

# ------------------------------
# (변경) Random Hard Negative
# ------------------------------
# neg_items_list = []
# pos_users_cpu = pos_users.cpu().tolist()

# for u in pos_users_cpu:
#     # 이 유저가 아직 안 본 아이템 중 하나를 랜덤으로 뽑는다.
#     neg_item = random.choice(unvisited_items[u])
#     neg_items_list.append(neg_item)

# neg_items = torch.tensor(neg_items_list, dtype=torch.long, device=device)
