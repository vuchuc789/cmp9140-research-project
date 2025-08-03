# from glob import glob
#
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def statistic():
    benign_df = pd.read_parquet("data_1/Benign.parquet.zst")
    anomalous_df = pd.read_parquet("data_1/Anomalous.parquet.zst")

    results = []
    for col in benign_df.columns:
        if col in [
            "Source IP",
            "Source Port",
            "Destination IP",
            "Destination Port",
            "Timestamp",
            "Protocol",
        ]:  # Skip non-feature columns
            continue
        try:
            benign_values = benign_df[col].dropna()
            attack_values = anomalous_df[col].dropna()

            # Choose a test (t-test or Mann-Whitney depending on distribution)
            stat, p_value = mannwhitneyu(
                benign_values, attack_values, alternative="two-sided"
            )

            results.append({"Feature": col, "p-value": p_value})
        except Exception as e:
            print(f"Skipped {col} due to: {e}")

    results_df = pd.DataFrame(results).sort_values("p-value")

    print(benign_df.info(verbose=True))
    print(results_df)
    print(results_df[results_df["p-value"] < 0.05])

    # benign_df_0 = pd.read_parquet("data_1/Benign.parquet.zst")
    # benign_df_1 = pd.read_parquet("data_0/Benign.parquet.zst")
    #
    # counts_0 = np.array([len(benign_df_0), 0])
    # for path in glob("data_1/*"):
    #     if not path.endswith(".parquet.zst") or "Benign" in path or "Anomalous" in path:
    #         continue
    #
    #     print(f"reading {path}...")
    #     df = pd.read_parquet(path)
    #     counts_0[1] += len(df)
    #
    # counts_1 = np.array([len(benign_df_1), 0])
    # for path in glob("data_0/*"):
    #     if not path.endswith(".parquet.zst") or "Benign" in path or "Anomalous" in path:
    #         continue
    #
    #     print(f"reading {path}...")
    #     df = pd.read_parquet(path)
    #     counts_1[1] += len(df)
    #
    # # Compute percentages
    # perc_a = counts_0 / counts_0.sum() * 100
    # perc_b = counts_1 / counts_1.sum() * 100
    #
    # # Setup
    # labels = ["CIC-IDS2017", "CIC-DDoS2019"]
    # x = np.arange(len(labels))  # [0, 1]
    # bar_width = 0.35
    #
    # fig, ax = plt.subplots()
    #
    # # Plot: left bar = benign, right bar = anomalous
    # benign_bars = ax.bar(
    #     x - bar_width / 2,
    #     perc_a[0:1].tolist() + perc_b[0:1].tolist(),
    #     bar_width,
    #     label="Benign",
    #     color="tab:blue",
    # )
    # anomalous_bars = ax.bar(
    #     x + bar_width / 2,
    #     perc_a[1:2].tolist() + perc_b[1:2].tolist(),
    #     bar_width,
    #     label="Anomalous",
    #     color="tab:orange",
    # )
    #
    # # Add raw counts as labels
    # ax.bar_label(
    #     benign_bars, labels=[f"{c:,}" for c in [counts_0[0], counts_1[0]]], fmt="%s"
    # )
    # ax.bar_label(
    #     anomalous_bars, labels=[f"{c:,}" for c in [counts_0[1], counts_1[1]]], fmt="%s"
    # )
    #
    # # Axes setup
    # ax.set_ylabel("Proportion (%)")
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # # ax.set_ylim(0, 100)
    # ax.set_title("Class Distribution by Dataset")
    # ax.legend()
    # ax.grid(True, linestyle="--", alpha=0.5)
    #
    # plt.tight_layout()
    # plt.show()
