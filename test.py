import torch
from sklearn import neighbors
import losses


def validate(src_model, tgt_model, src_data_loader, tgt_data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    with torch.no_grad():
        X, y = losses.extract_embeddings(src_model, src_data_loader)
        Xtest, ytest = losses.extract_embeddings(tgt_model, tgt_data_loader)

        clf = neighbors.KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        Xtest = Xtest
        ytest = ytest

        # Xtest = Xtest[:1000]
        # ytest = ytest[:1000]
        y_pred = clf.predict(Xtest)

        acc = (y_pred == ytest).mean()
        # print("Accuracy: {}".format(acc))

    return acc
