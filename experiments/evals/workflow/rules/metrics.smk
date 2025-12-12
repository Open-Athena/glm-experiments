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


rule aggregate_metrics:
    """Aggregate all metrics into long and wide format tables."""
    input:
        get_all_metric_files()
    output:
        wide="results/correlations/metrics_wide.parquet",
        long="results/correlations/metrics_long.parquet"
    run:
        import re
        from pathlib import Path

        records = []
        for file in input:
            # Parse: results/metrics/{dataset}/{metric}/{model}.tsv
            parts = Path(file).parts
            dataset = parts[2]
            metric_type = parts[3]
            model = parts[4].replace('.tsv', '')

            # Extract step and scoring from model name (e.g., "10000_LLR.minus.score")
            match = re.match(r'(\d+)_(.*)', model)
            if match:
                step = int(match.group(1))
                scoring = match.group(2)
            else:
                continue

            # Read metric value
            df = pd.read_csv(file, sep='\t')
            value = df.iloc[0, 0]

            records.append({
                'step': step,
                'dataset': dataset,
                'metric_type': metric_type,
                'scoring': scoring,
                'value': value
            })

        # Long format
        df_long = pd.DataFrame(records)
        df_long.to_parquet(output.long, index=False)

        # Wide format: pivot so each column is {dataset}_{metric_type}_{scoring}
        df_long['column_name'] = (
            df_long['dataset'] + '_' +
            df_long['metric_type'] + '_' +
            df_long['scoring']
        )
        df_wide = df_long.pivot(index='step', columns='column_name', values='value')
        df_wide = df_wide.reset_index()
        df_wide.to_parquet(output.wide, index=False)


rule compute_correlations:
    """Compute Pearson and Spearman correlation matrices."""
    input:
        "results/correlations/metrics_wide.parquet"
    output:
        pearson="results/correlations/pearson.tsv",
        spearman="results/correlations/spearman.tsv"
    run:
        df = pd.read_parquet(input[0])

        # Compute Pearson correlation
        pearson_corr = df.corr(method='pearson')
        pearson_corr.to_csv(output.pearson, sep='\t', float_format='%.3f')

        # Compute Spearman correlation
        spearman_corr = df.corr(method='spearman')
        spearman_corr.to_csv(output.spearman, sep='\t', float_format='%.3f')


rule plot_correlation_heatmaps:
    """Plot correlation matrices as annotated heatmaps."""
    input:
        pearson="results/correlations/pearson.tsv",
        spearman="results/correlations/spearman.tsv"
    output:
        pearson_png="results/correlations/pearson_heatmap.png",
        spearman_png="results/correlations/spearman_heatmap.png",
        pearson_pdf="results/correlations/pearson_heatmap.pdf",
        spearman_pdf="results/correlations/spearman_heatmap.pdf"
    run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Read correlation matrices
        pearson = pd.read_csv(input.pearson, sep='\t', index_col=0)
        spearman = pd.read_csv(input.spearman, sep='\t', index_col=0)

        # Reorder columns: step first, then by correlation with step (descending)
        step_corr_pearson = pearson.loc['step'].drop('step').sort_values(ascending=False)
        new_order = ['step'] + step_corr_pearson.index.tolist()
        pearson = pearson.loc[new_order, new_order]

        step_corr_spearman = spearman.loc['step'].drop('step').sort_values(ascending=False)
        new_order_spearman = ['step'] + step_corr_spearman.index.tolist()
        spearman = spearman.loc[new_order_spearman, new_order_spearman]

        # Plot Pearson
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(pearson, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax,
                    cbar_kws={'label': 'Pearson Correlation'},
                    linewidths=0.5, linecolor='gray')
        ax.set_title('Pearson Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(output.pearson_png, dpi=300, bbox_inches='tight')
        fig.savefig(output.pearson_pdf, bbox_inches='tight')
        plt.close()

        # Plot Spearman
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(spearman, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax,
                    cbar_kws={'label': 'Spearman Correlation'},
                    linewidths=0.5, linecolor='gray')
        ax.set_title('Spearman Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(output.spearman_png, dpi=300, bbox_inches='tight')
        fig.savefig(output.spearman_pdf, bbox_inches='tight')
        plt.close()


rule plot_metrics_vs_step:
    """Plot line plots showing how each metric changes with training step."""
    input:
        "results/correlations/metrics_long.parquet"
    output:
        png="results/correlations/metrics_vs_step.png",
        pdf="results/correlations/metrics_vs_step.pdf"
    run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_parquet(input[0])

        # Create faceted line plot
        df['label'] = df['dataset'] + '\n' + df['metric_type'] + '\n' + df['scoring']

        # Count unique combinations to determine grid size
        n_combos = df['label'].nunique()
        ncols = min(3, n_combos)
        nrows = (n_combos + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (label, group) in enumerate(df.groupby('label')):
            ax = axes[idx]
            group = group.sort_values('step')
            ax.plot(group['step'], group['value'], marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Training Step', fontsize=10)
            ax.set_ylabel('Metric Value', fontsize=10)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_combos, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        fig.savefig(output.png, dpi=300, bbox_inches='tight')
        fig.savefig(output.pdf, bbox_inches='tight')
        plt.close()


rule plot_metric_pairs:
    """Plot scatter plots for pairs of metrics to visualize correlations."""
    input:
        wide="results/correlations/metrics_wide.parquet",
        pearson="results/correlations/pearson.tsv"
    output:
        png="results/correlations/metric_pairs.png",
        pdf="results/correlations/metric_pairs.pdf"
    run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_parquet(input.wide)
        corr = pd.read_csv(input.pearson, sep='\t', index_col=0)

        # Get metric columns (exclude 'step')
        metric_cols = [col for col in df.columns if col != 'step']

        # Find top correlations (excluding self-correlations and duplicates)
        corr_pairs = []
        for i, col1 in enumerate(metric_cols):
            for j, col2 in enumerate(metric_cols):
                if i < j:  # Upper triangle only
                    corr_val = corr.loc[col1, col2]
                    corr_pairs.append((col1, col2, abs(corr_val), corr_val))

        # Sort by absolute correlation and take top 9
        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = corr_pairs[:min(9, len(corr_pairs))]

        # Create 3x3 grid
        ncols = 3
        nrows = (len(top_pairs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        for idx, (col1, col2, abs_corr, corr_val) in enumerate(top_pairs):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            # Scatter plot
            ax.scatter(df[col1], df[col2], alpha=0.6, s=100, edgecolors='black', linewidth=1)

            # Add regression line
            z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[col1].min(), df[col1].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(col1.replace('_', ' '), fontsize=9)
            ax.set_ylabel(col2.replace('_', ' '), fontsize=9)
            ax.set_title(f'r = {corr_val:.3f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(top_pairs), nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')

        plt.suptitle('Top Correlated Metric Pairs', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        fig.savefig(output.png, dpi=300, bbox_inches='tight')
        fig.savefig(output.pdf, bbox_inches='tight')
        plt.close()
