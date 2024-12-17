python3 main.py \
    --data_path /home/b.kabongo/datasets/TripAdvisor/data.csv \
    --train_size 0.8 \
    --val_size 0.1 \
    --test_size 0.1 \
    --seed 42 \
    --lr 0.001 \
    --n_epochs 50 \
    --batch_size 32 \
    --save_dir /home/b.kabongo/exps/TripAdvisor/PEPLER/ \
    --truncate_flag \
    --review_length 128 \
    --lower_flag \
    --delete_balise_flag \
    --delete_non_ascii_flag \