
def get_experiment_dict(args, exp_name):
        '''
        python main.py -e mnist2usps usps2mnist mnistBig2uspsBig uspsBig2mnistBig \
                          mnist2usps_nocenter usps2mnist_nocenter mnist2usps_nodisc \
                          usps2mnist_nodisc

        '''
        folds = str

        if exp_name == "coxs2v":
            exp_dict = {"still_dir": "/export/livia/data/lemoineh/COX-S2V/COX-S2V-Still-MTCNN160",
                        "video1_dir": "/export/livia/data/lemoineh/COX-S2V/COX-S2V-Video-MTCNN160/video2",
                        "video2_dir": "/export/livia/data/lemoineh/COX-S2V/COX-S2V-Video-MTCNN160/video4",
                        "video1_pairs": "dataset_utils/pair_files/coxs2v/video2_pairs.txt",
                        "video2_pairs": "dataset_utils/pair_files/coxs2v/video4_pairs.txt",
                        "cross_validation_num_fold": 10,
                        "image_size": 160,
                        "src_dataset": "coxs2v",
                        "src_model": "resnet18",
                        "src_epochs": 200,
                        "src_batch_size": 64,

                        "tgt_dataset": "coxs2v",
                        "tgt_model": "resnet18",
                        "tgt_epochs": 200,
                        "tgt_batch_size": 50,

                        "options": {"center": True, "disc": True},
                        "n_outputs": 128
                        }


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

        exp_dict["summary_path"] = ""

        return exp_dict