rule sat_mut_mpra_promoter_dataset:
    output:
        expand("results/dataset/sat_mut_mpra_promoter_{promoter}.parquet", promoter=config.sat_mut_mpra_promoter),
    run:
        V = pd.read_parquet("hf://datasets/gonzalobenegas/sat_mut_mpra/test.parquet")
        V["label"] = V["label"].abs()  # abs(LFC)
        for promoter, path in zip(config.sat_mut_mpra_promoter, output):
            V[V["element"] == promoter].to_parquet(path, index=False)
