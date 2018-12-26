import matplotlib
matplotlib.use('Agg')
import json
import torch
import misc as ms
import models
import datasets
import test
import os


def set_gpu(gpu_id):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu_id


def create_dirs(fname):
    if "/" not in fname:
        return

    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass


def save_json(fname, data):
    create_dirs(fname)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def load_json(fname):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d


def copy_models(exp_dict, path_dst):
    history = load_history(exp_dict)

    # src_model, src_opt = ms.load_model_src(exp_dict)
    # tgt_model, tgt_opt, disc, disc_opt  = ms.load_model_tgt(exp_dict)

    # create_dirs(path_dst + "/tmp")
    # torch.save(src_model.state_dict(), path_dst+"/model_src.pth")
    # torch.save(src_opt.state_dict(), path_dst+"/opt_src.pth")

    # torch.save(tgt_model.state_dict(), path_dst+"/model_tgt.pth")
    # torch.save(tgt_opt.state_dict(), path_dst+"/opt_tgt.pth")

    # torch.save(disc.state_dict(), path_dst+"/disc.pth")
    # torch.save(disc_opt.state_dict(), path_dst+"/disc_opt.pth")

    ms.save_json(path_dst + "/history.json", history)

    print("copied...")


def test_latest_model(exp_dict, verbose=1):

    history = load_history(exp_dict)

    src_trainloader, _ = ms.load_src_loaders(exp_dict)
    _, tgt_valloader = ms.load_tgt_loaders(exp_dict)

    src_model, src_opt = ms.load_model_src(exp_dict)
    tgt_model, tgt_opt, _, _ = ms.load_model_tgt(exp_dict)

    acc_tgt = test.validate(src_model, tgt_model, src_trainloader,
                            tgt_valloader)
    if verbose:
        print("====================="
              "\nAcc of model at epoch {}: {}\n"
              "=====================".format(history["tgt_train"][-1]["epoch"],
                                             acc_tgt))
    return acc_tgt


def load_src_loaders(exp_dict):
    train_loader = datasets.get_loader(
        exp_dict["src_dataset"],
        "train",
        batch_size=exp_dict["src_batch_size"])
    val_loader = datasets.get_loader(
        exp_dict["src_dataset"], "val", batch_size=exp_dict["src_batch_size"])
    n_train = len(train_loader.dataset)
    n_test = len(val_loader.dataset)
    name = type(train_loader.dataset).__name__

    print("Source ({}): train set: {} - val set: {}".format(
        name, n_train, n_test))
    return train_loader, val_loader


def load_tgt_loaders(exp_dict):
    train_loader = datasets.get_loader(
        exp_dict["tgt_dataset"],
        "train",
        batch_size=exp_dict["tgt_batch_size"])
    val_loader = datasets.get_loader(
        exp_dict["tgt_dataset"], "val", batch_size=exp_dict["tgt_batch_size"])
    name = type(train_loader.dataset).__name__
    n_train = len(train_loader.dataset)
    n_test = len(val_loader.dataset)
    print("Target ({}): train set: {} - val set: {}".format(
        name, n_train, n_test))
    return train_loader, val_loader


def load_history(exp_dict):
    name_history = exp_dict["path"] + "/history.json"

    if not os.path.exists(name_history) or (exp_dict["reset_src"]
                                            and exp_dict["reset_tgt"]):
        history = {"src_train": [{"epoch": 0}]}
        history["tgt_train"] = [{"epoch": 0, "acc_tgt": -1}]

        print("History from scratch...")
    else:
        history = ms.load_json(name_history)
        print("Loaded history {}".format(name_history))

    if exp_dict["reset_tgt"]:
        history["tgt_train"] = [{"epoch": 0, "acc_tgt": -1}]

        print("Resetting target training...")

    return history


def save_model_src(exp_dict, history, model_src, opt_src):
    save_json(exp_dict["path"] + "/history.json", history)
    torch.save(model_src.state_dict(), exp_dict["path"] + "/model_src.pth")
    torch.save(opt_src.state_dict(), exp_dict["path"] + "/opt_src.pth")
    print("Saved Source...")


def save_model_tgt(exp_dict, history, model_tgt, opt_tgt, disc, disc_opt):
    save_json(exp_dict["path"] + "/history.json", history)
    torch.save(model_tgt.state_dict(), exp_dict["path"] + "/model_tgt.pth")
    torch.save(opt_tgt.state_dict(), exp_dict["path"] + "/opt_tgt.pth")

    torch.save(disc.state_dict(), exp_dict["path"] + "/disc.pth")
    torch.save(disc_opt.state_dict(), exp_dict["path"] + "/disc_opt.pth")
    print("Saved Target...")


def load_model_src(exp_dict):
    src_model, src_opt = models.get_model(exp_dict["src_model"],
                                          exp_dict["n_outputs"])

    name_model = exp_dict["path"] + "/model_src.pth"
    name_opt = exp_dict["path"] + "/opt_src.pth"

    if os.path.exists(name_model) and not exp_dict["reset_src"]:
        src_model.load_state_dict(torch.load(name_model))
        src_opt.load_state_dict(torch.load(name_opt))
        print("Loading saved {}".format(name_model))

    else:
        print("Loading source models from scratch..")

    return src_model, src_opt


def load_model_tgt(exp_dict):
    tgt_model, tgt_opt = models.get_model(exp_dict["tgt_model"],
                                          exp_dict["n_outputs"])
    disc, disc_opt = models.get_model("disc", exp_dict["n_outputs"])

    name_model = exp_dict["path"] + "/model_tgt.pth"
    name_opt = exp_dict["path"] + "/opt_tgt.pth"

    name_disc = exp_dict["path"] + "/disc.pth"
    name_disc_opt = exp_dict["path"] + "/disc_opt.pth"

    if os.path.exists(name_model) and not exp_dict["reset_tgt"]:
        tgt_model.load_state_dict(torch.load(name_model))
        tgt_opt.load_state_dict(torch.load(name_opt))

        disc.load_state_dict(torch.load(name_disc))
        disc_opt.load_state_dict(torch.load(name_disc_opt))

        print("Loading saved {}".format(name_model))

    else:
        print("Loading target models from scratch..")

    return tgt_model, tgt_opt, disc, disc_opt
