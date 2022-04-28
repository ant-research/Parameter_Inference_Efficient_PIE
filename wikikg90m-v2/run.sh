SAVE_PATH=${2}
DATA_PATH=${1}

CUDA_VISIBLE_DEVICES=4,5,6,7  dglke_train --model_name TransE_l2 \
  --hidden_dim 600 --gamma 10  --valid --test  -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
  --gpu 0 1 2 3 \
  --async_update --no_save_emb \
  --print_on_screen --encoder_model_name shallow --save_path $SAVE_PATH \
  --data_path $DATA_PATH \
  --neg_sample_size 8192 --batch_size 8192 --lr 0.1 --regularization_coef 1.0e-9 \
  --max_step 20000000 --force_sync_interval 1000 --eval_interval 400000 \
  --LRE --LRE_rank 200
