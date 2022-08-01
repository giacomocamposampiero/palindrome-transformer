import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main(args):

    language = args.lan

    mod1 = "data/" + language + "1_models.csv"
    res1 = "data/" + language + "1_results.csv"
    mod2 = "data/" + language + "2_models.csv"
    res2 = "data/" + language + "2_results.csv"
    mod1sc = "data/" + language + "sc1_models.csv"
    res1sc = "data/" + language + "sc1_results.csv"
    mod2sc = "data/" + language + "sc2_models.csv"
    res2sc = "data/" + language + "sc2_results.csv"

    models1 = pd.read_csv(mod1)
    results1 = pd.read_csv(res1)
    data1 = pd.merge(models1, results1, on='runid')
    models2 = pd.read_csv(mod2)
    results2 = pd.read_csv(res2)
    data2 = pd.merge(models2, results2, on='runid')
    data = pd.concat((data1, data2))

    models1sc = pd.read_csv(mod1sc)
    results1sc = pd.read_csv(res1sc)
    data1sc = pd.merge(models1sc, results1sc, on='runid')
    models2sc = pd.read_csv(mod2sc)
    results2sc = pd.read_csv(res2sc)
    data2sc = pd.merge(models2sc, results2sc, on='runid')
    datasc = pd.concat((data1sc, data2sc))

    log = dict()
    gb = data[["train_length", "epoch", "trainloss", "vallos", "trainacc", "valacc"]].groupby(["train_length"])
    for name, group in gb:
        dat = group.groupby(["epoch"]).mean()
        log[name] = {
            'trainloss' : dat['trainloss'].values, 
            'vallos' : dat['vallos'].values, 
            'trainacc' : dat['trainacc'].values, 
            'valacc' : dat['valacc'].values
        }

    logsc = dict()
    gbsc = datasc[["train_length", "epoch", "trainloss", "vallos", "trainacc", "valacc"]].groupby(["train_length"])
    for name, group in gbsc:
        dat = group.groupby(["epoch"]).mean()
        logsc[name] = {
            'trainloss' : dat['trainloss'].values, 
            'vallos' : dat['vallos'].values, 
            'trainacc' : dat['trainacc'].values, 
            'valacc' : dat['valacc'].values
        }

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(14, 10)

    for len in log.keys():
        axs[0, 0].plot(range(0, 1000), log[len]['vallos'], label=len)
        axs[1, 0].plot(range(0, 1000), log[len]['valacc'], label=len)

    for len in logsc.keys():
        axs[0, 1].plot(range(0, 1000), logsc[len]['vallos'], label=len)
        axs[1, 1].plot(range(0, 1000), logsc[len]['valacc'], label=len)

    axs[0, 0 ].set_ylim(top=1.1, bottom=-0.1)
    axs[0, 0].set_ylabel("Validation loss (bit cross-entropy)",fontsize=16, labelpad=10)
    axs[1, 0 ].set_ylabel("Validation accuracy",fontsize=16, labelpad=10)
    axs[0, 0 ].set_ylim(top=1.1, bottom=-0.1)
    axs[0, 0 ].set_ylabel("Validation loss (bit cross-entropy)",fontsize=16, labelpad=10)
    axs[1, 0].set_ylabel("Validation accuracy",fontsize=16, labelpad=10)
    axs[0, 0].set_xscale("log")
    axs[1, 0].set_xscale("log")
    axs[1, 0].set_xlabel("Epochs",fontsize=16, labelpad=10)


    axs[1, 0 ].set_xlabel("Epochs",fontsize=16, labelpad=10)
    axs[1, 1].set_xlabel("Epochs",fontsize=16, labelpad=10)

    axs[0, 0].set_xscale("log")
    axs[1, 0].set_ylim(bottom=-0.1, top=1.1)
    axs[1, 0].set_xscale("log")
    axs[0, 1].set_xscale("log")
    axs[1, 1].set_ylim(bottom=-0.1, top=1.1)
    axs[1, 1].set_xscale("log")
    axs[1, 1].legend(loc='upper center', bbox_to_anchor=(-0.13, -0.2), fancybox=True, shadow=True, ncol=6)
    
    plt.savefig(language+"_generalize.pdf", transparent=True)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--lan', type=str, dest='lan', default='first', help='Possible values: ["first", "one"]')
    args = ap.parse_args()
    main(args)