python3 main.py \
    --dataset_name Hotels \
    --data_path /home/b.kabongo/aspects_datasets/Hotels/data.csv \
    --train_size 0.8 \
    --test_size 0.1 \
    --eval_size 0.1 \
    --embedding_dim 64 \
    --hidden_size 512 \
    --n_layers 2 \
    --dropout 0.2 \
    --review_length 128 \
    --vocab_size 20000 \
    --batch_size 64 \
    --lr 0.002 \
    --n_epochs 50 \
    --seed 42 \
    --verbose \
    --save_dir /home/b.kabongo/exps/Hotels/NRT/ \