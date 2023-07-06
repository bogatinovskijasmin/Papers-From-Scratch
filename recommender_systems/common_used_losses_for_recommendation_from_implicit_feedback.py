import torch

class BPRLoss(torch.nn.MSELoss):
    def __init__(self):
        super(BPRLoss, self).__init__()
    def forward(self, positive, negative):
        distances = positive - negative
        tmp = torch.log(torch.sigmoid(distances))
        loss = -torch.sum(tmp, dim=0, keepdim=True)
        return loss.mean()
class HingeLossbRec(torch.nn.HingeEmbeddingLoss):
    def __init__(self):
        super().__init__()
    def forward(self, positive, negative, margin):
        distances = positive - negative
        loss = torch.sum(torch.max(-distances + margin, 0))
        return loss
