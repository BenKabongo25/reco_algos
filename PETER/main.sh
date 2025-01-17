python3 main.py \
    --data_path /home/b.kabongo/aspects_datasets/Hotels/data.csv \
    --train_size 0.8 \
    --eval_size 0.1 \
    --test_size 0.1 \
    --lang en \
    --emsize 512 \
    --nhead 2 \
    --nhid 2048 \
    --nlayers 2 \
    --dropout 0.2 \
    --lr 1.0 \
    --clip 1.0 \
    --epochs 100 \
    --batch_size 32 \
    --seed 42 \
    --cuda \
    --log_interval 200 \
    --checkpoint /home/b.kabongo/exps/Hotels/PETER/ \
    --outf output.txt \
    --vocab_size 20000 \
    --endure_times 5 \
    --rating_reg 0.1 \
    --context_reg 1.0 \
    --text_reg 1.0 \
    --peter_mask \
    --no-use_feature \
    --words 128 \