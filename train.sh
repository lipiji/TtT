gamma=0.5
dname="HybirdSet"  # "SIGHAN15" "HybirdSet", "TtTSet"
dpath="./data/"$dname
bpath="./model/bert/"
cpath="./ckpt/"$dname"_"$gamma"/"

mkdir -p $cpath

python -u main.py \
    --bert_path  $bpath/bert.ckpt\
    --bert_vocab $bpath/vocab.txt \
    --train_data $dpath/train.txt \
    --dev_data $dpath/test.txt\
    --test_data $dpath/test.txt\
    --batch_size 100 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 1000 \
    --gpu_id 4 \
    --print_every 50 \
    --save_every 500 \
    --fine_tune \
    --loss_type FC_FT_CRF\
    --gamma $gamma \
    --model_save_path $cpath \
    --prediction_max_len 128 \
    --dev_eval_path $cpath/dev_pred.txt \
    --final_eval_path $cpath/dev_eval.txt \
    --l2_lambda 1e-5 \
    --training_max_len 128
