# Ben Kabongo
# December 2024

# Probabilistic Matrix Factorization
# Doc: https://cornac.readthedocs.io/en/stable/api_ref/models.html#module-cornac.models.pmf.recom_pmf


import argparse
import cornac
import json

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.common import main


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # PMF Cornac args
    args.add_argument("--k", type=int, default=10)
    args.add_argument("--max_iter", type=int, default=100)
    args.add_argument("--learning_rate", type=float, default=0.01)
    args.add_argument("--gamma", type=float, default=0.9)
    args.add_argument("--lambda_reg", type=float, default=0.001)
    args.add_argument("--variant", type=str, default="non_linear") # linear, non_linear
    args.add_argument("--trainable", action=argparse.BooleanOptionalAction)
    args.set_defaults(trainable=True)
    args.add_argument("--init_params_file", type=str, default="")
    args.add_argument("--seed", type=int, default=None)

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
    if not config.trainable:
        if config.init_params_file == "":
            raise ValueError("init_params_file must be provided when trainable=False")
        init_params = json.load(open(config.init_params_file, "r"))
    else:
        init_params = None

    model = cornac.models.PMF(
        k=config.k,
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        lambda_reg=config.lambda_reg,
        variant=config.variant,
        trainable=config.trainable,
        verbose=config.verbose,
        init_params=init_params,
        seed=config.seed
    )

    main(config, models=[model])
