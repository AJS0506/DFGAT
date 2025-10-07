python3 train_full.py --seed 1004 --dataset 0 --gpu 0 --emb-dim 64 --first-dim 32 --second-dim 16 &
python3 train_full.py --seed 1005 --dataset 0 --gpu 1 --emb-dim 64 --first-dim 32 --second-dim 16 &
python3 train_full.py --seed 1006 --dataset 0 --gpu 2 --emb-dim 64 --first-dim 32 --second-dim 16 &
python3 train_full.py --seed 1007 --dataset 0 --gpu 3 --emb-dim 64 --first-dim 32 --second-dim 16 &
python3 train_full.py --seed 1008 --dataset 0 --gpu -1 --emb-dim 64 --first-dim 32 --second-dim 16 &

python3 train_full.py --seed 1004 --dataset 3 --gpu 1 &
python3 train_full.py --seed 1005 --dataset 3 --gpu 2 &
python3 train_full.py --seed 1006 --dataset 3 --gpu 2 &
python3 train_full.py --seed 1007 --dataset 3 --gpu 3 &
python3 train_full.py --seed 1008 --dataset 3 --gpu -1 &

python3 train_full.py --seed 1004 --dataset 0 --gpu 1 &
python3 train_full.py --seed 1005 --dataset 0 --gpu 2 &
python3 train_full.py --seed 1006 --dataset 0 --gpu 2 &
python3 train_full.py --seed 1007 --dataset 0 --gpu 3 &
python3 train_full.py --seed 1008 --dataset 0 --gpu 1 &

---

python3 train_full.py --seed 1004 --dataset 1 --gpu 0 &
python3 train_full.py --seed 1005 --dataset 1 --gpu 1 &
python3 train_full.py --seed 1006 --dataset 1 --gpu 2 &
python3 train_full.py --seed 1007 --dataset 1 --gpu 3 &
python3 train_full.py --seed 1008 --dataset 1 --gpu 2 &

python3 train_full.py --seed 1004 --dataset 2 --gpu 0 &
python3 train_full.py --seed 1005 --dataset 2 --gpu 1 &
python3 train_full.py --seed 1006 --dataset 2 --gpu 2 &
python3 train_full.py --seed 1007 --dataset 2 --gpu 3 &
python3 train_full.py --seed 1008 --dataset 2 --gpu -1 &

python3 -u train_full.py --seed 1004 --dataset 1 --gpu 1 > MovieLens25M_1004.txt &
python3 -u train_full.py --seed 1005 --dataset 1 --gpu 2 > MovieLens25M_1005.txt &
python3 -u train_full.py --seed 1006 --dataset 1 --gpu 3 > MovieLens25M_1006.txt &
python3 -u train_full.py --seed 1007 --dataset 1 --gpu 2 > MovieLens25M_1007.txt &
python3 -u train_full.py --seed 1008 --dataset 1 --gpu 3 > MovieLens25M_1008.txt &

# pkill -f train_full.py
