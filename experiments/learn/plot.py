import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main(args):

    language = args.lan

    mod = "data/" + language + "_models.csv"
    res = "data/" + language + "_results.csv"

    models = pd.read_csv(mod)
    results = pd.read_csv(res)
    data = pd.merge(models, results, on='runid')

    log = dict()
    gb = data[["train_length", "epoch", "trainloss", "vallos", "trainacc", "valacc"]].groupby(["train_length"])
    for name, group in gb:
        data = group.groupby(["epoch"]).mean()
        log[name] = {
            'trainloss' : data['trainloss'].values, 
            'vallos' : data['vallos'].values, 
            'trainacc' : data['trainacc'].values, 
            'valacc' : data['valacc'].values
        }

    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(9, 10)

    for len in log.keys():
        ax1.plot(range(0, 100), log[len]['vallos'], label=len)
        ax2.plot(range(0, 100), log[len]['valacc'], label=len)

    if language != "palindrome" and language != "parity": ax1.set_ylim(top=1.1, bottom=-0.1)
    ax1.set_ylabel("Validation loss (bit cross-entropy)" , fontsize=16, labelpad=10)
    ax2.set_ylabel("Validation accuracy" , fontsize=16, labelpad=10)

    plt.xlabel("Epochs" , fontsize=16, labelpad=10)
    ax1.set_xscale("log")
    ax2.set_xscale("log")

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=6)
    # plt.legend(loc='center left', bbox_to_anchor=(1.01,1), prop={'size': 16})

    # plt.legend(bbox_to_anchor=(1.35, 1.5), title="String length.", fontsize=14)
    plt.tight_layout()
    plt.savefig(language+"_learn.pdf", transparent=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--lan', type=str, dest='lan', default='first')
    args = ap.parse_args()
    main(args)