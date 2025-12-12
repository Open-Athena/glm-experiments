rule metrics_AUPRC:
    input:
        "results/dataset/{dataset}.parquet",
        "results/prediction/{dataset}/{model}.parquet",
    output:
        "results/metrics/{dataset}/AUPRC/{model}.tsv",
    wildcard_constraints:
        dataset="|".join(get_all_datasets()),
    run:
        y_true = pd.read_parquet(input[0], columns=["label"]).label
        y_pred = pd.read_parquet(input[1], columns=["score"]).score
        AUPRC = average_precision_score(y_true, y_pred)
        pd.DataFrame({"AUPRC": [AUPRC]}).to_csv(output[0], sep="\t", index=False, float_format="%.3f")


rule metrics_AUROC:
    input:
        "results/dataset/{dataset}.parquet",
        "results/prediction/{dataset}/{model}.parquet",
    output:
        "results/metrics/{dataset}/AUROC/{model}.tsv",
    wildcard_constraints:
        dataset="|".join(get_all_datasets()),
    run:
        y_true = pd.read_parquet(input[0], columns=["label"]).label
        y_pred = pd.read_parquet(input[1], columns=["score"]).score
        AUROC = roc_auc_score(y_true, y_pred)
        pd.DataFrame({"AUROC": [AUROC]}).to_csv(output[0], sep="\t", index=False, float_format="%.3f")


rule metrics_Spearman:
    input:
        "results/dataset/{dataset}.parquet",
        "results/prediction/{dataset}/{model}.parquet",
    output:
        "results/metrics/{dataset}/Spearman/{model}.tsv",
    wildcard_constraints:
        dataset="|".join(get_all_datasets()),
    run:
        y_true = pd.read_parquet(input[0], columns=["label"]).label
        y_pred = pd.read_parquet(input[1], columns=["score"]).score
        Spearman = spearmanr(y_true, y_pred)[0]
        pd.DataFrame({"Spearman": [Spearman]}).to_csv(output[0], sep="\t", index=False, float_format="%.3f")
