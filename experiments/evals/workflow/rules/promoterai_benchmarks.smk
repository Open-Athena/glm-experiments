# PromoterAI benchmark datasets from GitHub
# Downloads and processes all benchmarks from: https://github.com/Illumina/PromoterAI/tree/master/data/benchmark

rule promoterai_benchmark:
    output:
        "results/dataset/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(config["promoterai_benchmarks"].keys()),
    params:
        filename=lambda wildcards: config["promoterai_benchmarks"][wildcards.dataset],
    run:
        url = f"https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/{params.filename}"
        V = pd.read_csv(url, sep="\t")
        V["chrom"] = V["chrom"].str.replace("^chr", "", regex=True)
        # Group by coordinates (variants near multiple genes)
        # Label is True if consequence is not "none" for ANY gene
        V_grouped = V.groupby(COORDINATES, as_index=False).agg({
            "consequence": lambda x: (x != "none").any(),
        })
        V_grouped = V_grouped.rename(columns={"consequence": "label"})
        V_grouped.to_parquet(output[0], index=False)
