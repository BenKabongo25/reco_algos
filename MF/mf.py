# Matrix Factorization
# Doc: https://cornac.readthedocs.io/en/stable/api_ref/models.html#module-cornac.models.mf.recom_mf


import argparse
import cornac
import json
import os
import pandas as pd


def main(config):

    # Data
    columns = [config.user_column, config.item_column, config.rating_column]
    timestamp = config.timestamp and config.fmt.lower().endswith("t")
    if timestamp:
        columns.append(config.timestamp_column)

    if config.data_path != "":
        data = pd.read_csv(config.data_path)[columns]
        data[config.rating_column] = data[config.rating_column].astype(float)

        if config.verbose:
            print("Data shape: ", data.shape)
            print(data.head())

        eval_method = cornac.eval_methods.RatioSplit(
            data=data.to_numpy(), test_size=config.test_size, val_size=config.val_size, 
            rating_threshold=config.rating_threshold, exclude_unknowns=config.exclude_unknowns, 
            seed=config.seed, verbose=config.verbose
        )

    else:
        train_data = pd.read_csv(config.train_path)[columns]
        train_data[config.rating_column] = train_data[config.rating_column].astype(float)
        test_data = pd.read_csv(config.test_path)[columns]
        test_data[config.rating_column] = test_data[config.rating_column].astype(float)
        val_data = None
        if config.val_path != "":
            val_data = pd.read_csv(config.val_path)[columns]
            val_data[config.rating_column] = val_data[config.rating_column].astype(float)

        if config.verbose:
            print("Train shape: ", train_data.shape)
            print(train_data.head())
            print("Test shape: ", test_data.shape)
            print(test_data.head())
            if val_data is not None:
                print("Val shape: ", val_data.shape)
                print(val_data.head())
            
        eval_method = cornac.eval_methods.BaseMethod.from_splits(
            train_data=train_data.to_numpy(), test_data=test_data.to_numpy(), 
            val_data=None if val_data is None else val_data.to_numpy(),
            fmt=config.fmt, 
            rating_threshold=config.rating_threshold, exclude_unknowns=config.exclude_unknowns, 
            seed=config.seed, verbose=config.verbose
        )

    # Model
    if not config.trainable:
        if config.init_params_file == "":
            raise ValueError("init_params_file must be provided when trainable=False")
        init_params = json.load(open(config.init_params_file, "r"))
    else:
        init_params = None
    
    model = cornac.models.MF(
        k=config.k,
        backend=config.backend,
        optimizer=config.optimizer,
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        lambda_reg=config.lambda_reg,
        dropout=config.dropout,
        use_bias=config.use_bias,
        early_stop=config.early_stop,
        num_threads=config.num_threads,
        trainable=config.trainable,
        verbose=config.verbose,
        init_params=init_params,
        seed=config.seed
    )

    # Evaluation metrics
    metrics = []
    if config.rmse:
        metrics.append(cornac.metrics.RMSE())
    if config.mae:
        metrics.append(cornac.metrics.MAE())
    if config.precision:
        metrics.append(cornac.metrics.Precision(k=config.ranking_k))
    if config.recall:
        metrics.append(cornac.metrics.Recall(k=config.ranking_k))
    if config.f1:    
        metrics.append(cornac.metrics.FMeasure(k=config.ranking_k))
    if config.auc:
        metrics.append(cornac.metrics.AUC())
    if config.ndcg:
        metrics.append(cornac.metrics.NDCG(k=config.ranking_k))
    if config.hit:
        metrics.append(cornac.metrics.HitRatio(k=config.ranking_k))
    if config.map:
        metrics.append(cornac.metrics.MAP())
    if config.mrr:
        metrics.append(cornac.metrics.MRR())

    # Training
    experiment = cornac.Experiment(
        eval_method=eval_method, models=[model], metrics=metrics,
        user_based=config.rating_user_based, show_validation=config.show_validation,
        verbose=config.verbose
    )
    experiment.run()
    test_results = experiment.result[0]
    val_results = None
    if config.val_path != "" or config.val_size > 0:
        val_results = experiment.val_result[0]
    
    # Results
    results = {"test": test_results.metric_avg_results, "val": val_results.metric_avg_results}
    os.makedirs(config.save_dir, exist_ok=True)
    json.dump(results, open(os.path.join(config.save_dir, f"{config.exp_name}.json"), "w"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # MF Cornac args
    args.add_argument("--k", type=int, default=10)
    args.add_argument("--backend", type=str, default="cpu") # cpu, pytorch
    args.add_argument("--optimizer", type=str, default="sgd") # adagrad, adam, rmsprop, sgd
    args.add_argument("--max_iter", type=int, default=100)
    args.add_argument("--learning_rate", type=float, default=0.01)
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--lambda_reg", type=float, default=0.001)
    args.add_argument("--dropout", type=float, default=0.0)
    args.add_argument("--use_bias", action=argparse.BooleanOptionalAction)
    args.set_defaults(use_bias=True)
    args.add_argument("--early_stop", action=argparse.BooleanOptionalAction)
    args.set_defaults(early_stop=True)
    args.add_argument("--num_threads", type=int, default=0)
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
    main(config)
