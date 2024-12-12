# Ben Kabongo
# December 2024

import cornac
import json
import pandas as pd
import os


def get_eval_method(config):
    columns = [config.user_column, config.item_column, config.rating_column]
    timestamp = config.timestamp and config.fmt.lower().endswith("t")
    if timestamp:
        columns.append(config.timestamp_column)

    if config.data_path != "":
        data = pd.read_csv(config.data_path)[columns]
        data[config.rating_column] = data[config.rating_column].astype(float)
        u_map = {u: i for i, u in enumerate(data[config.user_column].unique())}
        i_map = {i: j for j, i in enumerate(data[config.item_column].unique())}

        if config.verbose:
            print("Data shape: ", data.shape)
            print(data.head())

        eval_method = cornac.eval_methods.RatioSplit(
            data=data.to_numpy(), test_size=config.test_size, val_size=config.val_size, 
            global_uid_map=u_map, global_iid_map=i_map,
            fmt=config.fmt, rating_threshold=config.rating_threshold, 
            exclude_unknowns=config.exclude_unknowns, 
            seed=config.seed, verbose=config.verbose
        )

    else:
        train_data = pd.read_csv(config.train_path)[columns]
        train_data[config.rating_column] = train_data[config.rating_column].astype(float)
        test_data = pd.read_csv(config.test_path)[columns]
        test_data[config.rating_column] = test_data[config.rating_column].astype(float)
        data = pd.concat([train_data, test_data])

        val_data = None
        if config.val_path != "":
            val_data = pd.read_csv(config.val_path)[columns]
            val_data[config.rating_column] = val_data[config.rating_column].astype(float)
            data = pd.concat([data, val_data])

        u_map = {u: i for i, u in enumerate(data[config.user_column].unique())}
        i_map = {i: j for j, i in enumerate(data[config.item_column].unique())}

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
            global_uid_map=u_map, global_iid_map=i_map,
            fmt=config.fmt, rating_threshold=config.rating_threshold, 
            exclude_unknowns=config.exclude_unknowns, 
            seed=config.seed, verbose=config.verbose
        )
    return eval_method


def get_metrics(config):
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
    return metrics


def main(config, models):
    eval_method = get_eval_method(config)
    metrics = get_metrics(config)

    experiment = cornac.Experiment(
        eval_method=eval_method, models=models, metrics=metrics,
        user_based=config.rating_user_based, show_validation=config.show_validation,
        verbose=config.verbose
    )
    experiment.run()
    test_results = experiment.result
    val_results = None
    if config.val_path != "" or config.val_size > 0:
        val_results = experiment.val_result

    for i, model in enumerate(models):
        results = {"test": test_results[i].metric_avg_results, "val": val_results[i].metric_avg_results}
        os.makedirs(config.save_dir, exist_ok=True)
        json.dump(results, open(os.path.join(config.save_dir, f"{model.name}.json"), "w"))
