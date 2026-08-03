import torch
    
class SALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, labels):
        total_loss = 0.0
        total_items = 0

        for y, l in zip(y_pred, labels):
            mask = (l == 0)
            true_preds = y.masked_fill(mask, float('-inf'))

            golds_sum = torch.logsumexp(true_preds, dim=1)
            all_sum = torch.logsumexp(y, dim=1)

            total_loss += torch.sum(all_sum - golds_sum)
            total_items += y.shape[0]

        return total_loss / total_items
