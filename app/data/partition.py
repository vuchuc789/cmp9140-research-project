import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.data.data import DDoSDataset

# natural_160 = 10 partitions
# natural_80  = 20 partitions
# natural_32  = 50 partitions


def view_distribution(partition="none"):
    print("Loading partition 0...")
    first_partition = DDoSDataset(
        "data/Benign.parquet.zst",
        partition=partition,
        partition_id=0,
    )

    partitions = [first_partition]
    if partition != "none":
        for i in range(1, first_partition.partitioner.num_partitions):
            print(f"Loading partition {i}...")
            partitions.append(
                DDoSDataset(
                    "data/Benign.parquet.zst",
                    partition=partition,
                    partition_id=i,
                )
            )

    print(f"\nTotal: {first_partition.partitioner.num_partitions} partition(s)")

    df = pd.concat(
        [
            pd.DataFrame({"src_ip": p.partition_ids, "partition_id": pid})
            for pid, p in enumerate(partitions)
        ],
        ignore_index=True,
    )

    top_ips = df["src_ip"].value_counts().nlargest(9).index.tolist()
    df["src_ip"] = df["src_ip"].apply(lambda ip: ip if ip in top_ips else "others")

    # Create pivot table
    pivot = df.pivot_table(
        index="src_ip", columns="partition_id", aggfunc="size", fill_value=0
    )

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt="g",
        cmap="Greens",
        cbar_kws={"label": "Count"},
    )

    if len(partitions) >= 30:
        for text in ax.texts:
            text.set_rotation(90)
            text.set_verticalalignment("center")
            text.set_horizontalalignment("center")

    plt.title("Per Partition Source IP Distribution", fontsize=14)
    plt.xlabel("Partition ID", fontsize=12)
    plt.ylabel("Source IP", fontsize=12)

    plt.tight_layout()
    plt.show()

    # Transpose for plotting: columns → partitions
    pivot_T = pivot.T  # shape: (num_partitions, 20 groups)

    # Plot stacked bar chart
    pivot_T.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        colormap="Greens",  # You can also try 'tab20', 'viridis', etc.
    )

    plt.title("Stacked Source IP Distribution per Partition", fontsize=16)
    plt.xlabel("Partition ID", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.xticks(rotation=45)
    plt.legend(title="Source IP Group", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
