import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from sklearn import neighbors
import torch
import models
import losses

def vis_embed(src_model, tgt_model, src_loader, tgt_loader_eval):
    X,_ = losses.extract_embeddings(src_model, src_loader)
    
    Y,_ = losses.extract_embeddings(tgt_model, tgt_loader_eval)

    Y_labels = np.ones(Y.shape[0])*2
    X_labels = np.ones(X.shape[0])*1
    scatter(np.vstack([Y,X]), 
    np.hstack([Y_labels, X_labels]))

    # vis.scatter(X, X_labels)
    

def visTSN(src_model, tgt_model, src_loader, tgt_loader_eval):
    X,_ = losses.extract_embeddings(src_model, src_loader)
    
    Y,_ = losses.extract_embeddings(tgt_model, tgt_loader_eval)

    X = TSN(X)
    Y = TSN(Y)

    Y_labels = np.ones(Y.shape[0])*2
    X_labels = np.ones(X.shape[0])*1
    scatter(np.vstack([Y,X]), 
    np.hstack([Y_labels, X_labels]))


def visEmbed(exp_dict):
    src_loader = datasets.get_loader(exp_dict["src_dataset"], "train", 
                                         batch_size=exp_dict["src_batch_size"])
    
    tgt_val_loader =  datasets.get_loader(exp_dict["tgt_dataset"], "val", 
                                               batch_size=exp_dict["tgt_batch_size"])
        

    src_model, src_opt = models.get_model(exp_dict["src_model"], 
                                                        exp_dict["n_outputs"])
    src_model.load_state_dict(torch.load(exp_dict["path"]+"/model_src.pth"))

    tgt_model, tgt_opt = models.get_model(exp_dict["tgt_model"], 
                                                        exp_dict["n_outputs"])
    tgt_model.load_state_dict(torch.load(exp_dict["path"]+"/model_tgt.pth"))

    X,X_tgt = losses.extract_embeddings(src_model, src_loader)
    
    Y,Y_tgt = losses.extract_embeddings(tgt_model, tgt_val_loader)

    X, X_tgt = X[:500], X_tgt[:500]
    Y,Y_tgt = Y[:500], Y_tgt[:500]


    src_kmeans = KMeans(n_clusters=10)
    src_kmeans.fit(X)
    Xc = src_kmeans.cluster_centers_
 
    clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    clf.fit(X, X_tgt)
    Xc_tgt = clf.predict(Xc)
 




    # acc_tgt = test.validate(src_model, tgt_model, 
    #                                 src_loader, 
    #                                 tgt_val_loader)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #tsne.fit(Y[:500])
    S_tsne = tsne.fit_transform(np.vstack([Y,X,Xc]))
    #X_tsne = tsne.transform(X[:500])
    Y_tsne = S_tsne[:Y.shape[0]]
    X_tsne = S_tsne[Y.shape[0]:-10]
    Xc_tsne = S_tsne[-10:]
    # plt.mpl.rcParams['grid.color'] = 'k'
    # plt.mpl.rcParams['grid.linestyle'] = ':'
    # plt.mpl.rcParams['grid.linewidth'] = 0.5
    # Y_labels = Y_labels
    # X_labels = X_labels

    # scatter(Y_tsne, Y_tgt+1, win="1", title="target - {}".format(exp_dict["tgt_dataset"]))
    # scatter(X_tsne, X_tgt+1, win="2",title="source - {}".format(exp_dict["src_dataset"]))
           
    colors = ["b", "g", "r", "c", "m", "y","gray","w","chocolate","olive","pink"]

    if 1:
        fig = plt.figure(figsize=(6,6))
        plt.grid(linestyle='dotted')
        plt.scatter(X_tsne[:,0],X_tsne[:,1], alpha=0.6,edgecolors="black")
        

        for c in range(10):
            ind = Xc_tgt == c
            color=colors[c+1]
            plt.scatter(Xc_tsne[ind][:,0],Xc_tsne[ind][:,1],
                    s=250,c=color,edgecolors="black",marker="*")
        # plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel("t-SNE Feature 2")
        plt.ylabel("t-SNE Feature 1")
        title = "Source Dataset ({}) - Center: {} - Adv: {}".format(exp_dict["src_dataset"].upper().replace("BIG",""),
                        exp_dict["options"]["center"], exp_dict["options"]["disc"])
        plt.title(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("/mnt/home/issam/Summaries/src_{}.pdf".format(exp_dict["exp_name"].replace(" ","")),
                bbox_inches='tight', 
               transparent=False)

        plt.savefig("/mnt/home/issam/Summaries/src_{}.png".format(exp_dict["exp_name"]),
                bbox_inches='tight', 
               transparent=False)
        # ms.visplot(fig)


    if 1:

        fig = plt.figure(figsize=(6,6))
        plt.grid(linestyle='dotted')
        for c in range(10):
            ind = Y_tgt == c
            color=colors[c+1]
        
            plt.scatter(Y_tsne[ind][:,0],Y_tsne[ind][:,1], alpha=0.6,c=color,edgecolors="black")
        

        for c in range(10):
            ind = Xc_tgt == c
            color=colors[c+1]
            plt.scatter(Xc_tsne[ind][:,0],Xc_tsne[ind][:,1],
                    s=350,c=color,edgecolors="black",marker="*")
        # plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel("t-SNE Feature 2")
        plt.ylabel("t-SNE Feature 1")
        title = "Target Dataset ({}) - Center: {} - Adv: {}".format(exp_dict["tgt_dataset"].upper().replace("BIG",""),
                        exp_dict["options"]["center"], exp_dict["options"]["disc"])
        plt.title(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("/mnt/home/issam/Summaries/tgt_{}.pdf".format(exp_dict["exp_name"]),
                bbox_inches='tight', 
               transparent=False)

        plt.savefig("/mnt/home/issam/Summaries/tgt_{}.png".format(exp_dict["exp_name"]),
                bbox_inches='tight', 
               transparent=False)
        # ms.visplot(fig)
        
    # scatter(X_tsne, (X_tgt+1)/(X_tgt+1), win="2",title="source - {}".format(exp_dict["src_dataset"]))
    # scatter(Xc_tsne, Xc_tgt+1,append=True, win="2",markersymbol="star",
    #             markersize=20,title="source - {}".format(exp_dict["src_dataset"]))

    # Y_labels = np.ones(Y.shape[0])*2
    # X_labels = np.ones(X.shape[0])*1
    # scatter(np.vstack([Y_tsne[:,:2],X_tsne[:,:2]]), 
    #         np.hstack([Y_labels, X_labels]), win="3", title="both")

    # n_points = 1000
    # X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
    # n_neighbors = 10
    # n_components = 2

    # fig = plt.figure(figsize=(15, 8))
    # plt.suptitle("Manifold Learning with %i points, %i neighbors"
    #              % (1000, n_neighbors), fontsize=14)


    # t0 = time()
    # tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    # Y = tsne.fit_transform(X)
    # t1 = time()
    # print("t-SNE: %.2g sec" % (t1 - t0))
    # ax = fig.add_subplot(2, 5, 10)
    # plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    # plt.show()

import numpy as np

def plot(Y, X, line_name="", ylabel="", xlabel="", title="", 
         win="main", env="main"):
    import visdom

    vis = visdom.Visdom(port=1111)
    if not isinstance(Y, (list, np.ndarray)):
        Y = [Y]

    if not isinstance(X, (list, np.ndarray)):
        X = [X]

    if isinstance(Y, list):
        Y = np.array(Y)
    if isinstance(X, list):
        X = np.array(X)   

    msg = vis.updateTrace(Y=Y, X=X, name=line_name, env=env, win=win, 
                          append=True)

    if msg == 'win does not exist':
       options = dict(title=title , xlabel=xlabel, 
                      ylabel=ylabel, legend=[line_name])

       vis.line(X=X, Y=Y , opts=options, win=win, env=env) 



def scatter( X, Y=None, line_name="", ylabel="", xlabel="", title="", 
         win="main", env="main", markersymbol="dot", markersize=10, append=False):
    import visdom

    vis = visdom.Visdom(port=1111)


    if not isinstance(X, (list, np.ndarray)):
        X = [X]

    if isinstance(X, list):
        X = np.array(X)   




    options = dict(title=title , xlabel=xlabel, 
                  ylabel=ylabel,markersymbol=markersymbol,
                  markersize=markersize)
    if append:
        vis.scatter(X=X, Y=Y, win=win, env=env, name="twp", opts=options,update="insert")
    else:
        vis.scatter(X=X, Y=Y, win=win, env=env, opts=options)




def scatter_source(src_model, tgt_model, src_loader, 
        tgt_loader_eval, fname="p1"):
    X,_ = losses.extract_embeddings(src_model, src_loader)
    import pylab as plt
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0], X[:,1], s=8.0, color="C0")
    plt.grid(True)
    plt.rc('grid', linestyle="--", color='grey')
    plt.savefig("/mnt/home/issam/Summaries/{}.pdf".format(fname))
    plt.close()

def scatter_target(src_model, tgt_model, src_loader, 
    tgt_loader_eval, fname="p3"):
    X,_ = losses.extract_embeddings(src_model, src_loader)
    Y,_ = losses.extract_embeddings(tgt_model, tgt_loader_eval)
    import pylab as plt
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0], X[:,1], s=8.0, color="C0")
    plt.scatter(Y[:,0], Y[:,1], s=8.0, color="C1")
    plt.grid(True)
    plt.rc('grid', linestyle="--", color='grey')
    plt.savefig("/mnt/home/issam/Summaries/{}.pdf".format(fname))
    plt.close()
