import argparse
import torch
import random
import copy
import sys
import os
import datetime
import json

from collections import defaultdict
from pympler import asizeof

# ============= 모델 임포트 =============
from Model.MyGAT import DiffHeadGATRating

from GraphMaker.make_graph import GraphMaker
from DataLoader.load_dataset import DataLoader

from Eval.result import calcEvaluationScore
from Eval.tracker import TrainTracker

# =================== argparse 설정 =============================
parser = argparse.ArgumentParser(description="Optional arguments for DFGAT training.")
parser.add_argument("--gpu", type=int, default=-1, help="GPU ID (e.g., 0, 1, 2, 3) or -1 for CPU")
parser.add_argument("--dataset", type=int, default=3, help="Dataset index (0: movielens_small, 1: movielens_25M, 2: netflixPrize, 3: filmtrust)")
parser.add_argument("--seed", type=int, default=1008, help="Random seed")
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
dataset_list = ["movielens_small", "movielens_25M", "netflixPrize", "filmtrust"]
DATA_SET = dataset_list[DATASET_NUMBER]



# ============= 학습, 검증, 테스트 데이터셋 로드 =============
loader = DataLoader(DATA_SET)
train_set, val_set, test_set = loader.load_dataset()



# 메모리 사용량 계산
size_bytes = asizeof.asizeof(train_set)      # train_set과 모든 하위 객체 포함
size_mb    = size_bytes / (1024 ** 2)        # Byte → MB(2^20)

print(f"train_set 메모리 사용량: {size_mb:.2f} MB ({size_bytes:,} bytes)")
print("시드 -> ", RANDOM_SEED)
print("데이터셋 -> ", DATA_SET)


# ============= 데이터셋으로 그래프 만들기 =============
gm = GraphMaker()
graph = gm.get_graph(dataset=DATA_SET)
num_type1_nodes, num_type2_nodes = gm.get_num_nodes()

print("학습 그래프 -> ",graph)

# ============= TimeStamp Z‑정규화 =============

# 1) train_set에서 timestamp만 모아 float32 텐서로
ts_train = torch.tensor(
    [float(ts) for _, _, ts, _ in train_set],       # ← 반드시 float!
    dtype=torch.float32
)

# 2) 평균 (μ)·표준편차 (σ) 계산
μ = ts_train.mean()
σ = ts_train.std(unbiased=False).clamp_min(1e-8)   # 0 나눗셈 방지

def z_normalize(dataset, μ, σ):
    return [
        (src, dst, (float(ts) - μ.item()) / σ.item(), rt)
        for src, dst, ts, rt in dataset
    ]

# 3) train/val/test 모두 같은 Z‑스케일러 적용
train_set = z_normalize(train_set, μ, σ)
val_set   = z_normalize(val_set,   μ, σ)
test_set  = z_normalize(test_set,  μ, σ)

# ==============  노드별 Degree 데이터 처리  ===========
uid2dg = defaultdict(int)
mid2dg = defaultdict(int)
for uid, mid, ts, rating in train_set:
    uid2dg[uid] += 1
    mid2dg[mid] += 1 

# ============== 유저별 TimeStamp 데이터 처리 ==========
""" Key -> user ID, Val -> timestamp"""
uid2ts = defaultdict(list)
for uid, mid, ts, rating in train_set:
    uid2ts[uid].append(ts)






# ============== 유저별 Rating 데이터 처리 ==========
""" Key -> user ID, Val -> Ratings"""
uid2rt = defaultdict(list)
for uid, mid, ts, rating in train_set:
    uid2rt[uid].append(float(rating))

# ============== 아이템(영화)별 Rating 데이터 처리 ==========
""" Key -> item ID, Val -> Ratings"""
mid2rt = defaultdict(list)
for uid, mid, ts, rating in train_set:  
    mid2rt[mid].append(float(rating))



# ============== Rating Edge Score 추가 ==============
edge2rating = defaultdict(float)
for uid, mid, ts, rating in train_set:
    edge2rating[(uid, mid)] = float(rating) / 5.0  # 높은 평점에 큰 가중치
    edge2rating[(mid, uid)] = float(rating) / 5.0  # 높은 평점에 큰 가중치
    
    # 역방향 가중치: 낮은 평점에 더 큰 가중치 (negative signal 강조)
    # edge2rating[(uid, mid)] = (6.0 - float(rating)) / 5.0  # 1점→1.0, 5점→0.2
    # edge2rating[(mid, uid)] = (6.0 - float(rating)) / 5.0  # 1점→1.0, 5점→0.2

rt_weight_go, rt_weight_back = [], []

go_src, go_dst = graph.edges(etype="go")
back_src, back_dst = graph.edges(etype="back")

for g_src, g_dst in zip(go_src.tolist(), go_dst.tolist()):
    rt_weight_go.append(edge2rating[(g_src, g_dst)])

for b_src, b_dst in zip(back_src.tolist(), back_dst.tolist()):
    rt_weight_back.append(edge2rating[(b_src, b_dst)])

rt_weight_go = torch.tensor(rt_weight_go, dtype=torch.float32)
rt_weight_back = torch.tensor(rt_weight_back, dtype=torch.float32)


# ============= 학습 모델 초기화 및 GPU 설정 =============
device = torch.device("cpu") if GPU_ID == -1 else torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
graph = graph.to(device)

model = DiffHeadGATRating(
    num_user_nodes=num_type1_nodes,
    num_location_nodes=num_type2_nodes,
    emb_dim=embedding_dim,
    out1_dim=first_layer_dim,
    out2_dim=second_layer_dim,
    uid2ts=uid2ts,
    uid2dg=uid2dg,
    mid2dg=mid2dg,
    uid2rt=uid2rt,
    mid2rt=mid2rt,
    device=device,

    # ==== 정보주입 CUSTOM 인자 ===
    rt_weight_go = rt_weight_go,
    rt_weight_back = rt_weight_back
).to(device)

print(f">>> Using {'CPU' if GPU_ID == -1 else f'GPU: {GPU_ID}'}, Dataset: {DATASET_NUMBER}, Seed: {RANDOM_SEED}")


# ============= 옵티마이저 설정 =============
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


# ============= Hard Negative 샘플링을 위한 데이터셋 딕셔너리 =============
# 전체 방문 기록 (fallback용)
user_visited = defaultdict(set)
for uid, mid, _, _ in train_set + val_set + test_set:  # train_set만 사용 (data leakage 방지)
    user_visited[uid].add(mid)

# 1-2점 아이템 (Hard Negative용)
user_neg_visited = defaultdict(set)
for uid, mid, _, rt in train_set + val_set + test_set:
    if rt <= 2:
        user_neg_visited[uid].add(mid)


# GPT
seen_train = defaultdict(set)
for u, i, _, _ in train_set:   # explicit이면 평점 무관히 '본 것'은 전부 제외하는 게 일반적
    seen_train[u].add(i)

seen_val = defaultdict(set)
for u, i, _, _ in val_set:
    seen_val[u].add(i)




# ============= 에포크 및 배치사이즈 설정 =============
batch_size = len(train_set) // 30
print(f"훈련 세트 개수 -> {len(train_set)}, batch_size -> {batch_size}")


# ============= Loss Tracker =============
train_loss_tracker = []
val_loss_tracker = []


# ============= Early Stopping 구현 =============
early_stop_cnt = 0

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

        # pos_users = torch.LongTensor([row[0] for row in batch_data]).to(device)
        # pos_items = torch.LongTensor([row[1] for row in batch_data]).to(device)

        pos_users = torch.LongTensor([row[0] for row in batch_data if row[3] >= 3]).to(device)
        pos_items = torch.LongTensor([row[1] for row in batch_data if row[3] >= 3]).to(device)

        if len(pos_users) == 0:
            print('빈 배치가 있어 skip합니다.')
            continue

        # GCN 호출
        h = model(graph)
        user_emb = h['user']
        item_emb = h['item']

        # =========== Hard Negative Sampling (1-2점 우선) ===========
        neg_items_list = []
        pos_users_cpu = pos_users.cpu().tolist()  # 유저 ID를 CPU 리스트로 변환 (set/dict 접근 속도↑)

        for u in pos_users_cpu:
            if user_neg_visited[u]:  # 해당 유저가 1-2점 준 아이템이 있으면
                # Hard Negative: 실제로 낮게 평가한 아이템에서 샘플링
                neg_item = random.choice(list(user_neg_visited[u]))
                neg_items_list.append(neg_item)
            else:
                # 1-2점 아이템이 없으면 방문 안 한 아이템에서 랜덤 샘플링
                visited_set = user_visited[u] if u in user_visited else set()
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

            pos_users = torch.LongTensor([row[0] for row in val_batch if row[3] >= 3]).to(device)
            pos_items = torch.LongTensor([row[1] for row in val_batch if row[3] >= 3]).to(device)

            if len(pos_users) == 0:
                continue

            # 모델 순전파
            h = model(graph)
            user_emb = h['user']
            item_emb = h['item']

            # 학습시와 동일한 하드 네거티브 샘플링!
            neg_items_list = []
            pos_users_cpu = pos_users.cpu().tolist()

            for u in pos_users_cpu:
                if user_neg_visited[u]:  # 해당 유저가 1-2점 준 아이템이 있으면
                    # Hard Negative: 실제로 낮게 평가한 아이템에서 샘플링
                    neg_item = random.choice(list(user_neg_visited[u]))
                    neg_items_list.append(neg_item)
                else:
                    # 1-2점 아이템이 없으면 방문 안 한 아이템에서 랜덤 샘플링
                    visited_set = user_visited[u] if u in user_visited else set()
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

# 새로운 폴더 구조: BestModel/SEED/DATASET/MODEL_TIMESTAMP/
folder_path = f"BestModel/{RANDOM_SEED}/{DATA_SET}/DFGAT_{time_stamp}"

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
    "model_type": "DFGAT"
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

# test_processor = calcEvaluationScore(test_set, test_hidden, folder_path)
# avg_recall, avg_precision, avg_ndcg = test_processor.calcScore()

# tracker = TrainTracker(train_loss_tracker, val_loss_tracker, avg_recall, avg_precision, avg_ndcg, folder_path)
# tracker.plot_results()

# test_processor = calcEvaluationScore(test_set, test_hidden, folder_path, DATA_SET)
test_processor = calcEvaluationScore(
    dataset=test_set,
    hidden_state=test_hidden,
    folder_path=folder_path,
    dataset_name=DATA_SET,
    seen_train=seen_train,
    seen_val=seen_val,
    pos_threshold=4.0,
    exclude_seen_in_reco=True,
)

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



