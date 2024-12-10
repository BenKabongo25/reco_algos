python3 ncf.py \
    --data_path /home/b.kabongo/datasets/TripAdvisor/data.csv \
    --fmt UIR \
    --train_size 0.8 \
    --test_size 0.1 \
    --val_size 0.1 \
    --num_factors 16 \
    --layers 128 64 32 16 \
    --act_fn relu \
    --reg 0.0 \
    --num_epochs 50 \
    --batch_size 256 \
    --num_neg 4 \
    --lr 0.001 \
    --learner adam \
    --backend pytorch \
    --trainable \
    --seed 42 \
    --user_column user_id \
    --item_column item_id \
    --rating_column rating \
    --timestamp_column timestamp \
    --no-timestamp \
    --rating_threshold 4.0 \
    --no-exclude_unknowns \
    --rmse \
    --mae \
    --no-rating_user_based \
    --ranking_k 10 \
    --precision \
    --recall \
    --f1 \
    --auc \
    --ndcg \
    --hit \
    --map \
    --mrr \
    --exp_name MLP \
    --show_validation \
    --verbose \
    --save_dir /home/b.kabongo/exps/TripAdvisor/MLP/