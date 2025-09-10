from collections import defaultdict
from datetime import datetime

class NeflixPrizePreprocessor():

    # Netflix Prize 데이터셋은 train 셋과 test셋을 별도로 제공함. 이를 로드하여 사용
    def __init__(self):
        self.movieIdxRemapper = defaultdict(int)
        self.userIdxRemapper = defaultdict(int)
    
        self.full_edges = []
        self.edge_path = [f"./raw_data/combined_data_{i}.txt" for i in range(1,5)]


    # 1 데이터 로드
    def load_edges(self):
        
        cur_mid = None

        for file_path in self.edge_path:
            with open(file_path, 'r', encoding='utf8') as fp:
                for line in fp:
                    if ":" in line:
                        cur_mid = line.strip().replace(":","")
                    else:
                        uid, rating, timestamp = line.strip().split(",")
                        timestamp = self.transformUnixTime(timestamp)

                        if int(rating) >= 3:
                            self.full_edges.append([uid, cur_mid, timestamp, rating])
                        else:
                            continue


    # 타임스탬프 형식 변환
    def transformUnixTime(self, timestamp: str) -> int:
        # "YYYY-MM-DD" 형식으로 주어진 문자열을 datetime 객체로 변환
        dt = datetime.strptime(timestamp, "%Y-%m-%d")
        
        # datetime 객체를 유닉스 타임(1970-01-01 UTC 기준 초 단위)으로 변환
        # 단, tzinfo가 명시되어 있지 않으므로 시스템 로컬 타임존 기준!
        return int(dt.timestamp())


    def make_remap_dicts(self):
        # 유저/영화 ID를 담을 집합
        user_set = set()
        movie_set = set()

        # full_edges = [ [uid, mid, timestamp], ... ]
        # 1) 전체 유저, 영화 ID 수집
        for (uid, mid, timestamp, rating) in self.full_edges:
            user_set.add(uid)
            movie_set.add(mid)

        # 2) 정렬된 순서대로 딕셔너리 생성 (원본ID -> 0부터 매핑)
        for idx, original_uid in enumerate(sorted(user_set)):
            self.userIdxRemapper[original_uid] = idx
        
        for idx, original_mid in enumerate(sorted(movie_set)):
            self.movieIdxRemapper[original_mid] = idx
        
        # 3) full_edges에 적용 (uid, mid 부분만 교체)
        for i, (uid, mid, timestamp, rating) in enumerate(self.full_edges):
            remapped_uid = self.userIdxRemapper[uid]
            remapped_mid = self.movieIdxRemapper[mid]
   
            self.full_edges[i] = [remapped_uid, remapped_mid, timestamp, rating]


    def split_save_dat(self):

        TRAIN_RATIO = 0.6
        VAL_RATIO = 0.2
        TEST_RATIO = 0.2

        assert (TRAIN_RATIO + VAL_RATIO + TEST_RATIO) == 1.0

        uid2edges = defaultdict(list)
        
        train_edges, test_edges, val_edges = [], [], []

        for edge_data in self.full_edges:
            uid, mid, timestamp, rating = edge_data[0], edge_data[1], edge_data[2], edge_data[3]
            uid2edges[uid].append([mid, timestamp, rating])
        
        # 타임스탬프로 정렬
        for uid in uid2edges:
            uid2edges[uid].sort(key=lambda x:x[1])
        

        for uid in uid2edges:
            full_data = uid2edges[uid]

            if len(full_data) <= 5:
                print(f"5개 이하 데이터 무시함 uid -> {uid}")
                continue

            train_idx = int(len(full_data) * TRAIN_RATIO)
            val_idx   = int(len(full_data) * (TRAIN_RATIO + VAL_RATIO))
            
            # 훈련셋
            for mid, timestamp, rating in full_data[:train_idx]:
                train_edges.append([uid, mid, timestamp, rating])

            # 검증셋
            for mid, timestamp, rating in full_data[train_idx:val_idx]:
                val_edges.append([uid, mid, timestamp, rating])

            # 테스트셋
            for mid, timestamp, rating in full_data[val_idx:]:
                test_edges.append([uid, mid, timestamp, rating])


        # .dat 파일로 저장 (src::dst::timestamp::rating 형식)
        with open("netflixPrize_trn.dat", 'w', encoding='utf-8') as fp:
            for uid, mid, timestamp, rating in train_edges:
                fp.write(f"{uid}::{mid}::{timestamp}::{rating}\n")
        
        with open("netflixPrize_val.dat", 'w', encoding='utf-8') as fp:
            for uid, mid, timestamp, rating in val_edges:
                fp.write(f"{uid}::{mid}::{timestamp}::{rating}\n")
        
        with open("netflixPrize_tst.dat", 'w', encoding='utf-8') as fp:
            for uid, mid, timestamp, rating in test_edges:
                fp.write(f"{uid}::{mid}::{timestamp}::{rating}\n")
        
        return len(train_edges), len(val_edges), len(test_edges)

    def preprocess(self):
        self.load_edges()
        self.make_remap_dicts()
        train_count, val_count, test_count = self.split_save_dat()
        
        print("=== 전처리 완료! .dat 파일들이 생성되었습니다 ===")
        print(f"Train: {train_count} edges -> netflixPrize_trn.dat")
        print(f"Val: {val_count} edges -> netflixPrize_val.dat")
        print(f"Test: {test_count} edges -> netflixPrize_tst.dat")
        print("형식: src::dst::timestamp::rating")
        print("Note: rating >= 3만 포함")


if __name__ == "__main__" :
    pr = NeflixPrizePreprocessor()
    pr.preprocess()



        

    

    