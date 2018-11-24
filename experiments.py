
def get_experiment_dict(args, exp_name):
        '''
        python main.py -e mnist2usps usps2mnist mnistBig2uspsBig uspsBig2mnistBig \
                          mnist2usps_nocenter usps2mnist_nocenter mnist2usps_nodisc \
                          usps2mnist_nodisc

        '''

        if exp_name == "mnist2usps":
            exp_dict = {"src_dataset":"mnist",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "usps",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":50,

                        "options":{"center":True,"disc":True},
                        "n_outputs":256

                        }

        if exp_name == "usps2mnist":
            exp_dict = {"src_dataset":"usps",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "mnist",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":50,

                        "options":{"center":True,"disc":True},
                        "n_outputs":256



                        }

        if exp_name == "mnistBig2uspsBig":
            exp_dict = {"src_dataset":"mnistBig",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "uspsBig",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":50,

                        "options":{"center":True,"disc":True},
                        "n_outputs":256



                        }

        if exp_name == "uspsBig2mnistBig":
            exp_dict = {"src_dataset":"uspsBig",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "mnistBig",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":50,

                        "options":{
                                   "center":True,"disc":True
                                   },
                        "n_outputs":256



                        }
        if exp_name == "mnist2usps_nocenter":
            exp_dict = {"src_dataset":"mnist",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "usps",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":32,

                        "options":{
                                   "center":False,"disc":True
                                   },
                        "n_outputs":256



                        }

        if exp_name == "usps2mnist_nocenter":
            exp_dict = {"src_dataset":"usps",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "mnist",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":32,

                        "options":{
                                   "center":False,"disc":True
                                   },
                        "n_outputs":256
                        }


        if exp_name == "mnist2usps_nodisc":
            exp_dict = {"src_dataset":"mnist",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "usps",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":32,

                        "options":{
                                   "center":True,"disc":False,
                                   },
                        "n_outputs":256



                        }

        if exp_name == "mnistBig2uspsBig_nodisc":
            exp_dict = {"src_dataset":"mnistBig",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "uspsBig",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":32,

                        "options":{
                                   "center":True,"disc":False,
                                   },
                        "n_outputs":256



                        }

        if exp_name == "mnistBig2uspsBig_nocenter":
            exp_dict = {"src_dataset":"mnistBig",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "uspsBig",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":32,

                        "options":{
                                   "center":True,"disc":False,
                                   },
                        "n_outputs":256



                        }

        if exp_name == "usps2mnist_nodisc":
            exp_dict = {"src_dataset":"usps",
                        "src_model": "lenet",
                        "src_epochs":200,
                        "src_batch_size":64,

                        "tgt_dataset": "mnist",
                        "tgt_model": "lenet",
                        "tgt_epochs":200,
                        "tgt_batch_size":32,

                        "options":{
                                   "center":True,"disc":False,
                                   },
                        "n_outputs":256
                        }

        exp_dict["exp_name"] = exp_name

        exp_dict["path"]="checkpoints/{}/".format(exp_name)

        # exp_dict["path"]="/mnt/home/issam/Saves/" \
        #                     "DA/{}_{}_{}_{}_{}_{}_{}_{}/".format(exp_dict["src_dataset"], 
        #                                              exp_dict["src_model"],
        #                                              exp_dict["src_batch_size"],
        #                                              exp_dict["n_outputs"],
        #                                              exp_dict["tgt_dataset"], 
        #                                              exp_dict["tgt_model"],
        #                                              exp_dict["tgt_batch_size"],
        #                                              str(exp_dict["options"]).replace(" ",""))
        exp_dict["summary_path"] = "/mnt/home/issam/Summaries/"

        return exp_dict