
import torch
import torch.nn.functional as F
import numpy as np

from itertools import combinations

def triplet_loss(model, batch):
    model.train()
    emb = model(batch["X"].cuda())
    y = batch["y"].cuda()
    

    with torch.no_grad():
        triplets = get_triplets(emb, y)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + 1.)

    return losses.mean()


def center_loss(tgt_model, batch, src_model, src_centers, tgt_centers, 
                src_kmeans, tgt_kmeans, margin=1):
    # triplets = self.triplet_selector.get_triplets(embeddings, target, embeddings_adv=embeddings_adv)
    # triplets = triplets.cuda()


    #f_N = embeddings_adv[triplets[:, 2]]

    f_N_clf = tgt_model.forward(batch["X"].cuda()).view(batch["X"].shape[0], -1) #AttributeError: 'ResNet' object has no attribute 'convnet'

    #f_N = tgt_model.fc(f_N_clf.detach())
    f_N = f_N_clf
    #est.predict(f_N.cpu().numpy())
    y_src = src_kmeans.predict(f_N.detach().cpu().numpy())
    #ap_distances = (emb_centers[None] - f_N[:,None]).pow(2).min(1)[0].sum(1)
    ap_distances = (src_centers[y_src] - f_N).pow(2).sum(1)
    #ap_distances = (f_C[None] - f_N[:,None]).pow(2).sum(1).sum(1)

    
    #an_distances = 0
    losses = ap_distances.mean()

    # y_tgt = tgt_kmeans.predict(f_N.detach().cpu().numpy())
    # ap_distances = (tgt_centers[y_tgt] - f_N).pow(2).max(1)[0]

    # losses += ap_distances.mean()*0.1

    # f_P = src_model(batch["X"].cuda())
    #an_distances = (f_P - f_N).pow(2).sum(1)
    #losses -= an_distances.mean() * 0.1
  
    return losses



### Triplets Utils

def extract_embeddings(model, dataloader):
    model.eval()
    batch_size = dataloader.batch_sampler.num_batches
    n_samples = batch_size * len(dataloader)
    n_outputs = model.last.bias.shape[0]
    embeddings = np.zeros((n_samples, n_outputs))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            if k%10000==0:
                print(k)
            images = images.cuda()            
            embeddings[k:k+len(images)] = model.forward(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    return embeddings, labels
    
def get_triplets(embeddings, y):

  margin = 1
  D = pdist(embeddings)
  D = D.cpu()

  y = y.cpu().data.numpy().ravel()
  trip = []

  for label in set(y):
      label_mask = (y == label)
      label_indices = np.where(label_mask)[0]
      if len(label_indices) < 2:
          continue
      neg_ind = np.where(np.logical_not(label_mask))[0]
      
      ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
      ap = np.array(ap)

      ap_D = D[ap[:, 0], ap[:, 1]]
      
      # # GET HARD NEGATIVE
      # if np.random.rand() < 0.5:
      #   trip += get_neg_hard(neg_ind, hardest_negative,
      #                D, ap, ap_D, margin)
      # else:
      trip += get_neg_hard(neg_ind, random_neg,
                 D, ap, ap_D, margin)

  if len(trip) == 0:
      ap = ap[0]
      trip.append([ap[0], ap[1], neg_ind[0]])

  trip = np.array(trip)

  return torch.LongTensor(trip)




def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors)) 
    D += vectors.pow(2).sum(dim=1).view(1, -1) 
    D += vectors.pow(2).sum(dim=1).view(-1, 1)

    return D


def get_neg_hard(neg_ind, 
                      select_func,
                      D, ap, ap_D, margin):
    trip = []

    for ap_i, ap_di in zip(ap, ap_D):
        loss_values = (ap_di - 
               D[torch.LongTensor(np.array([ap_i[0]])), 
                torch.LongTensor(neg_ind)] + margin)

        loss_values = loss_values.data.cpu().numpy()
        neg_hard = select_func(loss_values)

        if neg_hard is not None:
            neg_hard = neg_ind[neg_hard]
            trip.append([ap_i[0], ap_i[1], neg_hard])

    return trip

def random_neg(loss_values):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards) if len(neg_hards) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def semihard_negative(loss_values, margin=1):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None