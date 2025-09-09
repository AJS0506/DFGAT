
import os

class DataLoader():
    def __init__(self, dataset:str):
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
        # 현재 파일의 위치에서 C_Dataset 폴더 경로 계산
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        base_path = os.path.join(project_root, "C_Dataset")
        
        self.train_file_path = os.path.join(base_path, f"{dataset}_trn.dat")
        self.val_file_path = os.path.join(base_path, f"{dataset}_val.dat")
        self.test_file_path = os.path.join(base_path, f"{dataset}_tst.dat")

    def load_dat_file(self, filepath):
        """DAT 파일을 읽어서 리스트로 변환"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) == 3:  # Gowalla (rating 없음)
                    uid = int(parts[0])
                    iid = int(parts[1])
                    timestamp = int(parts[2])
                    data.append([uid, iid, timestamp])
                elif len(parts) == 4:  # 나머지 데이터셋 (rating 있음)
                    uid = int(parts[0])
                    iid = int(parts[1])
                    timestamp = int(parts[2])
                    # rating은 문자열로 유지 (원본 PKL 형식 유지)
                    rating = parts[3]
                    data.append([uid, iid, timestamp, rating])
        return data

    def load_dataset(self):
        
        self.train_set = self.load_dat_file(self.train_file_path)
        self.val_set = self.load_dat_file(self.val_file_path)
        self.test_set = self.load_dat_file(self.test_file_path)

        return (self.train_set, self.val_set, self.test_set)
   