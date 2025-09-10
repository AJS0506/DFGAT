import os
from collections import defaultdict

class FilTrustPreprocessor:
    def __init__(self):
        # [유저, 영화, 타임스탬프(가짜), 평점] 형태로 저장할 리스트
        self.edge_datas = []

        # ID를 0부터 매핑하기 위한 사전
        self.userIdxRemapper = defaultdict(int)
        self.movieIdxRemapper = defaultdict(int)

        # ratings.txt 파일 위치 - raw_data 디렉터리에서 읽기
        self.ratings_path = "./raw_data/ratings.txt"
    
    def load_edge_data(self):
        """
        ratings.txt 파일을 읽어와서
        [uid, mid, timestamp, rating] 형태로 self.edge_datas에 저장

        파일 예시 (헤더 없음):
        1 1 2
        1 2 4
        ...
        각 라인은 (userID, movieID, rating) 순서로 되어 있음.
        timestamp는 실제로 없으므로, i번째 줄을 가짜 timestamp로 사용.
        """
        with open(self.ratings_path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                # 공백으로 구분된 세 개의 값: 유저ID, 영화ID, 평점
                user_str, movie_str, rating_str = line.split()
                
                user = user_str.strip()   # 문자열 상태 그대로 저장
                movie = movie_str.strip() # 문자열 상태 그대로 저장
                rating = float(rating_str)

                # i번째 줄을 timestamp(가짜)로 지정
                timestamp = i

                # Movielens 형식과 동일하게 [uid, mid, timestamp, rating]
                self.edge_datas.append([user, movie, timestamp, rating])

    def make_remapper_dict(self):
        """
        1) 전체 데이터에서 유저와 영화 ID를 각각 모은 뒤,
        2) 정수 기준 정렬(key=int)하여 0부터 매핑
        3) self.edge_datas에 있는 user, movie 컬럼을 remapper 사전의 값으로 치환
        """
        user_set = set()
        movie_set = set()

        # 유저와 영화 ID 수집
        for edge_data in self.edge_datas:
            user_set.add(edge_data[0])   # uid(문자열)
            movie_set.add(edge_data[1])  # mid(문자열)

        # 정수형으로 정렬
        for idx, original_uid in enumerate(sorted(user_set, key=int)):
            self.userIdxRemapper[original_uid] = idx
        for idx, original_mid in enumerate(sorted(movie_set, key=int)):
            self.movieIdxRemapper[original_mid] = idx

        # edge_datas에 반영 (문자열 → 매핑된 int)
        for edge_data in self.edge_datas:
            # edge_data = [uid, mid, timestamp, rating]
            edge_data[0] = self.userIdxRemapper[edge_data[0]]  # uid 매핑
            edge_data[1] = self.movieIdxRemapper[edge_data[1]] # mid 매핑

    def split_save_dat(self):
        """
        - 유저별로 데이터를 모은 뒤, timestamp 순으로 정렬
        - 60% / 20% / 20% 비율로 (train, val, test) 분할
        - 유저별 평점 수가 5개 이하인 경우는 무시
        - .dat 파일로 src::dst::timestamp::rating 형식으로 저장
        """
        TRAIN_RATIO = 0.6
        VAL_RATIO   = 0.2
        TEST_RATIO  = 0.2
        assert (TRAIN_RATIO + VAL_RATIO + TEST_RATIO) == 1.0

        # uid에 따른 시청 기록을 모은다: uid2edges[uid] = [[timestamp, rating, mid], ...]
        uid2edges = defaultdict(list)

        for edge_data in self.edge_datas:
            # edge_data = [uid, mid, timestamp, rating]
            uid, mid, timestamp, rating = edge_data
            # uid별로 [timestamp, rating, mid] 묶어서 저장
            uid2edges[uid].append([timestamp, rating, mid])
        
        # timestamp 기준 정렬
        for uid in uid2edges:
            # uid2edges[uid] = [[timestamp, rating, mid], ...]
            # 여기서 timestamp는 인덱스 0
            uid2edges[uid].sort(key=lambda x: x[0])

        train_edges = []
        val_edges   = []
        test_edges  = []

        # 사용자별로 Train/Val/Test 분할
        for uid, record_list in uid2edges.items():
            # 만약 평점 횟수가 5개 이하인 경우 무시
            if len(record_list) <= 5:
                print(f"유저 {uid}는 평점 기록이 5개 이하이므로 스킵합니다.")
                continue

            total_count = len(record_list)
            train_end = int(total_count * TRAIN_RATIO)
            val_end   = int(total_count * (TRAIN_RATIO + VAL_RATIO))

            # 실제로 저장할 때는 Movielens 형식 [uid, mid, timestamp, rating]으로 다시 맞춘다
            # record = [timestamp, rating, mid]
            for (timestamp, rating, mid) in record_list[:train_end]:
                train_edges.append([uid, mid, timestamp, rating])
            for (timestamp, rating, mid) in record_list[train_end:val_end]:
                val_edges.append([uid, mid, timestamp, rating])
            for (timestamp, rating, mid) in record_list[val_end:]:
                test_edges.append([uid, mid, timestamp, rating])

        # .dat 파일로 저장 (src::dst::timestamp::rating 형식)
        with open("filmtrust_trn.dat", 'w', encoding='utf-8') as fp:
            for uid, mid, timestamp, rating in train_edges:
                fp.write(f"{uid}::{mid}::{timestamp}::{rating}\n")
        
        with open("filmtrust_val.dat", 'w', encoding='utf-8') as fp:
            for uid, mid, timestamp, rating in val_edges:
                fp.write(f"{uid}::{mid}::{timestamp}::{rating}\n")
                
        with open("filmtrust_tst.dat", 'w', encoding='utf-8') as fp:
            for uid, mid, timestamp, rating in test_edges:
                fp.write(f"{uid}::{mid}::{timestamp}::{rating}\n")
        
        return len(train_edges), len(val_edges), len(test_edges)

    def preprocess(self):
        """
        전체 전처리 단계 순서:
        1) ratings.txt 로드 -> edge_datas ([uid, mid, timestamp, rating])
        2) uid, mid를 0부터 매핑 (key=int 정렬)
        3) timestamp 기준으로 Train/Val/Test 분할 및 저장
        """
        self.load_edge_data()
        self.make_remapper_dict()
        train_count, val_count, test_count = self.split_save_dat()

        print("=== 전처리 완료! .dat 파일들이 생성되었습니다 ===")
        print(f"Train: {train_count} edges -> filmtrust_trn.dat")
        print(f"Val: {val_count} edges -> filmtrust_val.dat") 
        print(f"Test: {test_count} edges -> filmtrust_tst.dat")
        print("형식: src::dst::timestamp::rating")


if __name__ == "__main__":
    pr = FilTrustPreprocessor()
    pr.preprocess()
