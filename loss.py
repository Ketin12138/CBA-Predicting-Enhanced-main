import torch
import torch.nn as nn

def supervised_contrastive_loss(embeddings, labels, epsilon=1e-8):
    Ns = (labels == 1).nonzero(as_tuple=True)[0]  
    Nb = (labels == 0).nonzero(as_tuple=True)[0]  

    if len(Ns) == 0 or len(Nb) == 0:
        return torch.tensor(0.0, requires_grad=True)

    pos_sim = torch.matmul(embeddings[Ns], embeddings[Ns].t())  
    neg_sim = torch.matmul(embeddings[Ns], embeddings[Nb].t())  

    pos_exp = torch.exp(pos_sim.clamp(max=10))  
    neg_exp = torch.exp(neg_sim.clamp(max=10))

    loss = -torch.log(pos_exp.sum(dim=1) / (pos_exp.sum(dim=1) + neg_exp.sum(dim=1) + epsilon))
    return loss.mean()

def compute_total_loss(output, embeddings, labels, lambda_contra=10.0):
    criterion_bce = nn.BCEWithLogitsLoss()
    ce_loss = criterion_bce(output.squeeze(), labels)
    contra_loss = supervised_contrastive_loss(embeddings, labels)
    return ce_loss + lambda_contra * contra_loss