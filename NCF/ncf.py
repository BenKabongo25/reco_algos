# Ben Kabongo
# December 2024

# NCF: Neural Collaborative Filtering
# Doc: https://cornac.readthedocs.io/en/stable/api_ref/models.html#module-cornac.models.ncf.recom_neumf


import argparse
import cornac

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.cornac_common import main


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # NCF Cornac args
    args.add_argument("--num_factors", type=int, default=8)
    args.add_argument("--layers", type=int, nargs="+", default=[64, 32, 16, 8])
    args.add_argument("--act_fn", type=str, default="relu") # ‘sigmoid’, ‘tanh’, ‘elu’, ‘relu’, ‘selu, ‘relu6’, ‘leaky_relu’
    args.add_argument("--reg", type=float, default=0.0)
    args.add_argument("--num_epochs", type=int, default=20)
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--num_neg", type=int, default=4)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--learner", type=str, default="adam") # adagrad, adam, rmsprop, sgd
    args.add_argument("--backend", type=str, default="pytorch") # tensorflow, pytorch
    args.add_argument("--trainable", action=argparse.BooleanOptionalAction)
    args.set_defaults(trainable=True)
    args.add_argument("--seed", type=int, default=42)

    # Data args
    args.add_argument("--data_path", type=str, default="")

    args.add_argument("--train_path", type=str, default="")
    args.add_argument("--val_path", type=str, default="")
    args.add_argument("--test_path", type=str, default="")    
    
    args.add_argument("--train_size", type=float, default=0.8)
    args.add_argument("--test_size", type=float, default=0.1)
    args.add_argument("--val_size", type=float, default=0.1)

    args.add_argument("--fmt", type=str, default="UIRT")
    args.add_argument("--user_column", type=str, default="user_id")
    args.add_argument("--item_column", type=str, default="item_id")
    args.add_argument("--rating_column", type=str, default="rating")
    args.add_argument("--timestamp_column", type=str, default="timestamp")
    args.add_argument("--timestamp", action=argparse.BooleanOptionalAction)
    args.set_defaults(timestamp=False)
    
    args.add_argument("--binary", action=argparse.BooleanOptionalAction)
    args.set_defaults(binary=False)
    args.add_argument("--rating_threshold", type=float, default=4.0)
    args.add_argument("--exclude_unknowns", action=argparse.BooleanOptionalAction)
    args.set_defaults(exclude_unknowns=False)

    # Metrics
    args.add_argument("--rmse", action=argparse.BooleanOptionalAction)
    args.set_defaults(rmse=True)
    args.add_argument("--mae", action=argparse.BooleanOptionalAction)
    args.set_defaults(mae=True)
    args.add_argument("--rating_user_based", action=argparse.BooleanOptionalAction)
    args.set_defaults(rating_user_based=False)

    args.add_argument("--ranking_k", type=int, default=10)
    args.add_argument("--precision", action=argparse.BooleanOptionalAction)
    args.set_defaults(precision=False)
    args.add_argument("--recall", action=argparse.BooleanOptionalAction)
    args.set_defaults(recall=False)
    args.add_argument("--f1", action=argparse.BooleanOptionalAction)
    args.set_defaults(f1=False)
    args.add_argument("--auc", action=argparse.BooleanOptionalAction)
    args.set_defaults(auc=False)
    args.add_argument("--ndcg", action=argparse.BooleanOptionalAction)
    args.set_defaults(ndcg=False)
    args.add_argument("--hit", action=argparse.BooleanOptionalAction)
    args.set_defaults(hit=False)
    args.add_argument("--map", action=argparse.BooleanOptionalAction)
    args.set_defaults(map=False)
    args.add_argument("--mrr", action=argparse.BooleanOptionalAction)
    args.set_defaults(mrr=False)

    # Experiment
    args.add_argument("--exp_name", type=str, default="MF")
    args.add_argument("--show_validation", action=argparse.BooleanOptionalAction)
    args.set_defaults(show_validation=False)
    args.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    args.set_defaults(verbose=True)
    args.add_argument("--save_dir", type=str, default="")

    config = args.parse_args()

    # Model
    if config.binary:
        NeuMF = cornac.models.NeuMF
    else:
        from regression_pytorch import NeuMFRecommender
        NeuMF = NeuMFRecommender

    model = NeuMF(
        num_factors=config.num_factors,
        layers=config.layers,
        act_fn=config.act_fn,
        reg=config.reg,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        num_neg=config.num_neg,
        lr=config.lr,
        learner=config.learner, 
        backend=config.backend,
        trainable=config.trainable,
        verbose=config.verbose,
        seed=config.seed
    )

    main(config, models=[model])
