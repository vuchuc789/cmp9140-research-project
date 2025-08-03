import matplotlib.pyplot as plt
import numpy as np


def loss():
    print("Loading training result...\n")
    with open("model/fedavg_iid_50_ae_history.npy", "rb") as f:
        avg_loss = np.load(f)[:30]
        _ = np.load(f)
        _ = np.load(f)
        avg_auc = np.load(f)[:30]
    with open("model/fedadam_fedprox_iid_50_ae_history.npy", "rb") as f:
        adam_loss = np.load(f)[:30]
        _ = np.load(f)
        _ = np.load(f)
        adam_auc = np.load(f)[:30]
    with open("model/fedyogi_iid_50_ae_history.npy", "rb") as f:
        yogi_loss = np.load(f)[:30]
        _ = np.load(f)
        _ = np.load(f)
        yogi_auc = np.load(f)[:30]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot training and validation losses
    ax1.plot(
        avg_loss,
        label="FedAvg Loss",
        color="tomato",
    )
    ax1.plot(
        adam_loss,
        label="FedAdam Loss",
        color="limegreen",
    )
    ax1.plot(
        yogi_loss,
        label="FedYogi Loss",
        color="royalblue",
    )
    # ax1.plot(
    #     batched_benign_test_loss,
    #     label="Validation Loss",
    #     color="tomato",
    #     linewidth=2,
    #     linestyle="--",
    # )
    # ax1.plot(
    #     batched_anomalous_test_loss,
    #     label="Anomalous Loss",
    #     color="crimson",
    #     linewidth=2,
    #     linestyle=":",
    # )

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Loss (MSE)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Second Y-axis for AUC
    ax2 = ax1.twinx()
    ax2.plot(
        avg_auc,
        label="FedAvg AUC",
        color="tomato",
        linestyle="--",
    )
    ax2.plot(
        adam_auc,
        label="FedAdam AUC",
        color="limegreen",
        linestyle="--",
    )
    ax2.plot(
        yogi_auc,
        label="FedYogi AUC",
        color="royalblue",
        linestyle="--",
    )
    ax2.set_ylabel("AUC", fontsize=12)
    ax2.set_ylim(0.0, 1.05)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    plt.title(
        "Federated Optimizer Comparision",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()
