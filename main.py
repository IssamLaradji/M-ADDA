"""Main script for ADDA."""
import sys
sys.path.append("../_EXTRAS")

import matplotlib
matplotlib.use('Agg')
import torch

import argparse
import pandas as pd
import numpy as np
import misc as ms
import experiments
from addons import vis
from addons import pretty_plot
import train

#per vedere se salva

if __name__ == '__main__':

    # SEE IF CUDA IS AVAILABLE
    assert torch.cuda.is_available()
    print("CUDA: %s" % torch.version.cuda)
    print("Pytroch: %s" % torch.__version__)
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--expList', nargs="+", default=None)
    parser.add_argument('-b', '--borgy', default=0, type=int)
    parser.add_argument('-br', '--borgy_running', default=0, type=int)
    parser.add_argument('-m', '--mode', default="max")
    parser.add_argument('-rs', '--reset_src', default=0, type=int)
    parser.add_argument('-rt', '--reset_tgt', default=0, type=int)
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-s', '--summary', type=int, default=0)
    parser.add_argument('-c', '--configList', nargs="+", default=None)
    parser.add_argument('-l', '--lossList', nargs="+", default=None)
    parser.add_argument('-d', '--datasetList', nargs="+", default=None)
    parser.add_argument('-metric', '--metricList', nargs="+", default=None)
    parser.add_argument('-model', '--modelList', nargs="+", default=None)

    args = parser.parse_args()
    ms.set_gpu(args.gpu)

    # init random seed
    # init_random_seed(10)
    results = {}

    pp_main = pretty_plot.PrettyPlot(
        ratio=0.5,
        figsize=(5, 4),
        legend_type="line",
        yscale="linear",
        subplots=(1, 1),
        shareRowLabel=True)
    for exp_name in args.expList:
        exp_dict = experiments.get_experiment_dict(args, exp_name)
        exp_dict["reset_src"] = args.reset_src
        exp_dict["reset_tgt"] = args.reset_tgt

        # SET SEED
        np.random.seed(10)
        torch.manual_seed(10)
        torch.cuda.manual_seed_all(10)

        history = ms.load_history(exp_dict)


        # Main options
        if args.mode == "test_model":
            results[exp_name] = ms.test_latest_model(exp_dict, verbose=0)

        elif args.mode == "train":
            train.train(exp_dict)


        if args.mode == "copy_models":
            results[exp_name] = ms.copy_models(
                exp_dict, path_dst="{}/".format(exp_name))

        # MISC
        if args.mode == "plot_src":

            src_losses = np.array(pd.DataFrame(history["src_train"])["loss"])
            src_epochs = np.array(pd.DataFrame(history["src_train"])["epoch"])

            pp_main.add_yxList(
                y_vals=src_losses[1:101],
                x_vals=src_epochs[1:101],
                label=exp_name.split("2")[0].upper().replace("BIG", ""),
                converged=None)

        if args.mode == "plot_tgt":

            tgt_acc = np.array(pd.DataFrame(history["tgt_train"])["acc_tgt"])
            src_epochs = np.array(pd.DataFrame(history["tgt_train"])["epoch"])

            pp_main.add_yxList(
                y_vals=tgt_acc[1:101],
                x_vals=src_epochs[1:101],
                label=exp_name.split("2")[1].upper().replace("BIG", ""),
                converged=None)
            # vis.visEmbed(exp_dict)

        if args.mode == "vis":
            vis.visEmbed(exp_dict)

        elif args.mode == "summary":

            summary = pd.DataFrame(history["tgt_train"][1:])["acc_tgt"]

            print(summary.describe())

        elif args.mode == "acc_tgt":

            summary = pd.DataFrame(history["tgt_train"][1:200])["acc_tgt"]

            print(summary)

        elif args.mode == "max":
            try:

                summary = pd.DataFrame(history["tgt_train"][1:])["acc_tgt"]
                results[exp_name] = summary.max()
            except:
                print("{} skipped...".format(exp_name))

    print(pd.Series(results))
    # Train Source
    if args.mode == "plot_src":
        pp_main.plot(ylabel="Triplet Loss", xlabel="Epochs", yscale="log")

        path = exp_dict["summary_path"]
        pp_main.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        figName = "%s/png_plots/SRC_%s.png" % (path, exp_name)
        ms.create_dirs(figName)
        pp_main.fig.savefig(figName)

        pp_main.fig.tight_layout()
        pp_main.fig.suptitle("")

        figName = "%s/pdf_plots/SRC_%s.pdf" % (path, exp_name)
        ms.create_dirs(figName)
        pp_main.fig.savefig(figName, dpi=600)

        print("saved {}".format(figName))

    if args.mode == "plot_tgt":
        pp_main.plot(
            ylabel="Classifcation Accuracy", xlabel="Epochs", yscale="log")

        path = exp_dict["summary_path"]
        pp_main.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        figName = "%s/png_plots/TGT_%s.png" % (path, exp_name)
        ms.create_dirs(figName)
        pp_main.fig.savefig(figName)

        pp_main.fig.tight_layout()
        pp_main.fig.suptitle("")

        figName = "%s/pdf_plots/TGT_%s.pdf" % (path, exp_name)
        ms.create_dirs(figName)
        pp_main.fig.savefig(figName, dpi=600)

        print("saved {}".format(figName))

