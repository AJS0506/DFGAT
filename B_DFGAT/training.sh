python3 train_ratingset.py --seed 1004 --dataset 0 --gpu 0 &
python3 train_ratingset.py --seed 1005 --dataset 0 --gpu 1 &
python3 train_ratingset.py --seed 1006 --dataset 0 --gpu 2 &
python3 train_ratingset.py --seed 1007 --dataset 0 --gpu 3 &
python3 train_ratingset.py --seed 1008 --dataset 0 --gpu -1 &


python3 train_ratingset.py --seed 1004 --dataset 3 --gpu 0 &
python3 train_ratingset.py --seed 1005 --dataset 3 --gpu 1 &
python3 train_ratingset.py --seed 1006 --dataset 3 --gpu 2 &
python3 train_ratingset.py --seed 1007 --dataset 3 --gpu 3 &
python3 train_ratingset.py --seed 1008 --dataset 3 --gpu -1 &

# pkill -f train_ratingset.py
