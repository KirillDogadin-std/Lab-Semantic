import torch


def relation_logistic_loss(phs, prs, pts, nhs, nrs, nts):
    pos_distance = phs + prs - pts
    neg_distance = nhs + nrs - nts
    pos_score = -torch.sum(torch.square(pos_distance), dim=1)
    neg_score = -torch.sum(torch.square(neg_distance), dim=1)
    pos_loss = torch.sum(torch.log(1 + torch.exp(-pos_score)))
    neg_loss = torch.sum(torch.log(1 + torch.exp(neg_score)))
    loss = torch.add(pos_loss, neg_loss)
    return loss


def logistic_loss_wo_negs(phs, pas, pvs, pws):
    pos_distance = phs + pas - pvs
    pos_score = -torch.sum(torch.square(pos_distance), dim=1)
    pos_score = torch.log(1 + torch.exp(-pos_score))
    pos_score = torch.multiply(pos_score, pws)
    pos_loss = torch.sum(pos_score)
    return pos_loss


def relation_logistic_loss_wo_negs(phs, prs, pts):
    pos_distance = phs + prs - pts
    pos_score = -torch.sum(torch.square(pos_distance), dim=1)
    pos_loss = torch.sum(torch.log(1 + torch.exp(-pos_score)))
    return pos_loss
