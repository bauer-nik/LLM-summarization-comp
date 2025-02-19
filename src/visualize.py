from matplotlib import pyplot as plt
import seaborn as sns

def graph_fmeasure(df, rouge, save_graph=False, output_path=None):
    sns.boxplot(data=df, y=f"{rouge}_fmeasure", hue="model")
    plt.title(f"Average fmeasure for {rouge} Compared Across Model")
    plt.xlabel("model")
    plt.ylabel(f"{rouge} fmeasure")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    if save_graph and output_path is not None:
        plt.savefig(f"{output_path}/boxplot_{rouge}_fmeasure.png", bbox_inches="tight")


def graph_precision_and_recall(df, rouge, save_graph=False, output_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    sns.boxplot(data=df, y=f"{rouge}_precision", hue="model", ax=ax1)
    ax1.set_title(f"Average Precision for {rouge} Compared Across Model")
    ax1.set_xlabel("model")
    ax1.set_ylabel(f"{rouge} precision")
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.get_legend().remove()

    sns.boxplot(data=df, y=f"{rouge}_recall", hue="model", ax=ax2)
    ax2.set_title(f"Average Recall for {rouge} Compared Across Model")
    ax2.set_xlabel("model")
    ax2.set_ylabel(f"{rouge} recall")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

    if save_graph and output_path is not None:
        plt.savefig(f"{output_path}/boxplot_{rouge}_precision_recall.png", bbox_inches="tight")
