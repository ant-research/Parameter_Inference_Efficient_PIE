
DATA_PATH=${1}

python -u get_candidate_dgl.py --dataset 'wikikg90M' \
   --data_path ${DATA_PATH}  --test \
   --batch_size_eval 1 --num_hops 3 \
   --num_proc 15  \
   --save_file 'test_candidate'


python -u get_candidate_dgl.py --dataset 'wikikg90M' \
   --data_path ${DATA_PATH}  --valid \
   --batch_size_eval 1 --num_hops 3 \
   --num_proc 15  \
   --save_file 'valid_candidate'

