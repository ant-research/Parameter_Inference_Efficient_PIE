GPU=${1}
DATA=${2}
CUDA_VISIBLE_DEVICES=${GPU} python -u ./src/main.py --cuda \
  --context_hops 3 --batch_size 512 --lr 0.0001 --dim 1024 --l2 0.0  --steps 10000000 \
  --cpu_num 1  --dataset WikiKG90Mv2 --neighbor_samples 6 \
  --neighbor_agg mean \
  --data_path ${DATA} \
  --use_ranking_loss True --margin 3 --gamma 3 

