import datetime
from collections import defaultdict

class GowallaPreprocessor():
    def __init__(self):
        # 유저 - 지역 간의 엣지
        self.hetero_edges = []
        # 유저 - 유저 간의 엣지 -> Gowalla 데이터셋에만 제공됨.
        self.user_edges = []

        self.checkin_path = "./raw_data/Gowalla_totalCheckins.txt"
        self.user_edges_path = "./raw_data/Gowalla_edges.txt"

        # remap 딕셔너리
        self.userIdxRemapper = {}
        self.locationIdxRemapper = {}

    def transfromUnixTime(self, checkinTime:str) -> int:
        dt = datetime.datetime.strptime(checkinTime, "%Y-%m-%dT%H:%M:%SZ")
        unix_time = int(dt.timestamp()) # 1287530127.0 ( float ) -> int
        
        return unix_time


    def load_hetero_edges(self):
        """
            [user]	    [check-in time]		  [latitude]	            [longitude]    [location id]
            0	2010-10-19T23:55:27Z	30.2359091167	-97.7951395833	22847
            0	2010-10-18T22:17:43Z	30.2691029532	-97.7493953705	420315
        """
        with open(self.checkin_path,'r', encoding='utf8') as fp:
            next(fp) # 헤더 건너뜀

            for line in fp:
                data = line.strip().split("\t")
                uid, timestamp, latitude, longitude, lid = data[0], data[1], data[2], data[3], data[4]
                timestamp = self.transfromUnixTime(timestamp)

                self.hetero_edges.append([uid, lid, timestamp])

    def load_user_edges(self):
        """
            0	1
            0	2
            0	3
            0	4
            0	5
        """
        with open(self.user_edges_path, 'r', encoding='utf8') as fp:
            for line in fp:
                data = line.strip().split("\t")
                src, dst = data[0], data[1]
                self.user_edges.append([src, dst])

    
    def make_remap_dicts(self):
        """
        userID, locationID를 0부터 순차적으로 매핑하는 딕셔너리 생성 및 적용
        """
        user_set = set()
        location_set = set()

        # 1) hetero_edges에서 user, location 전부 수집
        for (uid, lid, _) in self.hetero_edges:
            user_set.add(uid)
            location_set.add(lid)

        # 2) user_edges에서 user만 추가로 수집
        for (src, dst) in self.user_edges:
            user_set.add(src)
            user_set.add(dst)

        # 3) Remap 딕셔너리 생성 (정렬된 순서대로 0부터 할당)
        for idx, original_uid in enumerate(sorted(user_set)):
            self.userIdxRemapper[original_uid] = idx

        for idx, original_lid in enumerate(sorted(location_set)):
            self.locationIdxRemapper[original_lid] = idx

        # 4) hetero_edges에 적용 (user, location)
        for i, (uid, lid, timestamp) in enumerate(self.hetero_edges):
            self.hetero_edges[i][0] = self.userIdxRemapper[uid]
            self.hetero_edges[i][1] = self.locationIdxRemapper[lid]

        # 5) user_edges에 적용 (user, user)
        for i, (src, dst) in enumerate(self.user_edges):
            self.user_edges[i][0] = self.userIdxRemapper[src]
            self.user_edges[i][1] = self.userIdxRemapper[dst]


    def split_save_dat(self):
        TRAIN_RATIO = 0.6
        VAL_RATIO = 0.2
        TEST_RATIO = 0.2

        assert (TRAIN_RATIO + VAL_RATIO + TEST_RATIO) == 1.0

        uid2edges = defaultdict(list)
        train_edges, val_edges, test_edges = [], [], []

        for edge_data in self.hetero_edges:
            uid, lid, timestamp = edge_data[0], edge_data[1], edge_data[2]
            uid2edges[uid].append([lid, timestamp])

            
        for uid in uid2edges:
            uid2edges[uid].sort(key=lambda x:x[-1])

        for uid in uid2edges:
            full_data = uid2edges[uid]
            
            if len(full_data) <= 5:
                print(f"5개 이하 데이터 무시함 uid -> {uid}")
                continue

            train_idx = int(len(full_data) * TRAIN_RATIO)
            val_idx   = int(len(full_data) * (TRAIN_RATIO + VAL_RATIO))

            # 훈련셋
            for mid, timestamp in full_data[:train_idx]:
                train_edges.append([uid, mid, timestamp])

            # 검증셋
            for mid, timestamp in full_data[train_idx:val_idx]:
                val_edges.append([uid, mid, timestamp])

            # 테스트셋
            for mid, timestamp in full_data[val_idx:]:
                test_edges.append([uid, mid, timestamp])

        # .dat 파일로 저장 (src::dst::timestamp 형식)
        with open("gowalla_trn.dat", 'w', encoding='utf-8') as fp:
            for uid, lid, timestamp in train_edges:
                fp.write(f"{uid}::{lid}::{timestamp}\n")
        
        with open("gowalla_val.dat", 'w', encoding='utf-8') as fp:
            for uid, lid, timestamp in val_edges:
                fp.write(f"{uid}::{lid}::{timestamp}\n")

        with open("gowalla_tst.dat", 'w', encoding='utf-8') as fp:
            for uid, lid, timestamp in test_edges:
                fp.write(f"{uid}::{lid}::{timestamp}\n")
        
        # 유저-유저 엣지도 .dat 파일로 저장
        with open("gowalla_homoedge.dat", 'w', encoding='utf-8') as fp:
            for src, dst in self.user_edges:
                fp.write(f"{src}::{dst}\n")
        
        return len(train_edges), len(val_edges), len(test_edges)

    def save_dat(self):
        with open("edges_gowalla.dat", 'w', encoding='utf-8') as fp:
            for uid, lid, timestamp in self.hetero_edges:
                fp.write(f"{uid}::{lid}::{timestamp}\n")

        with open("edges_gowalla_user.dat", 'w', encoding='utf-8') as fp:
            for src, dst in self.user_edges:
                fp.write(f"{src}::{dst}\n")

    def preprocess(self):
        self.load_hetero_edges()
        self.load_user_edges()
        self.make_remap_dicts()
        # self.save_dat()
        train_count, val_count, test_count = self.split_save_dat()
        
        print("=== 전처리 완료! .dat 파일들이 생성되었습니다 ===")
        print(f"Train: {train_count} edges -> gowalla_trn.dat")
        print(f"Val: {val_count} edges -> gowalla_val.dat")
        print(f"Test: {test_count} edges -> gowalla_tst.dat")
        print(f"User-User edges: {len(self.user_edges)} -> gowalla_homoedge.dat")
        print("형식: src::dst::timestamp (rating 없음)")

if __name__ == "__main__":
    pr = GowallaPreprocessor()
    pr.preprocess()
