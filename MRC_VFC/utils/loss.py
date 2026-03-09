import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer
from torch.autograd import Variable


class ProbabilityLoss(nn.Module):
    def __init__(self):
        super(ProbabilityLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, logits1, logits2):
        assert logits1.size() == logits2.size()
        softmax1 = self.softmax(logits1)
        softmax2 = self.softmax(logits2)

        probability_loss = self.criterion(softmax1.log(), softmax2)
        return probability_loss


class BatchLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(BatchLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, activations, ema_activations):
        assert activations.size() == ema_activations.size()
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            activations = torch.cat(GatherLayer.apply(activations), dim=0)
            ema_activations = torch.cat(GatherLayer.apply(ema_activations), dim=0)
        # reshape as N*C
        activations = activations.view(N, -1)
        ema_activations = ema_activations.view(N, -1)

        # form N*N similarity matrix
        similarity = activations.mm(activations.t())
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        ema_similarity = ema_activations.mm(ema_activations.t())
        ema_norm = torch.norm(ema_similarity, 2, 1).view(-1, 1)
        ema_similarity = ema_similarity / ema_norm

        batch_loss = (similarity - ema_similarity) ** 2 / N
        return batch_loss


class ChannelLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(ChannelLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, activations, ema_activations):
        assert activations.size() == ema_activations.size()
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            activations = torch.cat(GatherLayer.apply(activations), dim=0)
            ema_activations = torch.cat(GatherLayer.apply(ema_activations), dim=0)
        # reshape as N*C
        activations = activations.view(N, -1)
        ema_activations = ema_activations.view(N, -1)

        # form C*C channel-wise similarity matrix
        similarity = activations.t().mm(activations)
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        ema_similarity = ema_activations.t().mm(ema_activations)
        ema_norm = torch.norm(ema_similarity, 2, 1).view(-1, 1)
        ema_similarity = ema_similarity / ema_norm

        channel_loss = (similarity - ema_similarity) ** 2 / N
        return channel_loss


class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class GaussianPriorLoss(nn.Module):
    """
    Class-conditional Gaussian prior with diagonal covariance.
    Maintains EMA stats for each class and penalizes NLL under the class Gaussian.
    """
    def __init__(self, num_classes, ema_momentum=0.1, var_floor=1e-4, mode="nll", fixed_var_value=1.0):
        super(GaussianPriorLoss, self).__init__()
        self.num_classes = num_classes
        self.ema_momentum = ema_momentum
        self.var_floor = var_floor
        self.mode = mode
        self.fixed_var_value = fixed_var_value
        self.means = None
        self.vars = None

    def _lazy_init(self, feat_dim, device, dtype):
        self.means = torch.zeros(self.num_classes, feat_dim, device=device, dtype=dtype)
        self.vars = torch.ones(self.num_classes, feat_dim, device=device, dtype=dtype)

    def forward(self, features, labels):
        if self.means is None or self.vars is None:
            self._lazy_init(features.size(1), features.device, features.dtype)

        # Update EMA stats for classes present in the batch (no grad)
        with torch.no_grad():
            for cls in labels.unique():
                cls = int(cls.item())
                mask = labels == cls
                if mask.sum() < 2:
                    continue
                cls_feats = features[mask]
                batch_mean = cls_feats.mean(dim=0)
                batch_var = cls_feats.var(dim=0, unbiased=False).clamp_min(self.var_floor)
                self.means[cls] = (1 - self.ema_momentum) * self.means[cls] + self.ema_momentum * batch_mean
                self.vars[cls] = (1 - self.ema_momentum) * self.vars[cls] + self.ema_momentum * batch_var

        means = self.means[labels]
        diff = features - means
        if self.mode == "center":
            return (diff * diff).mean(dim=1).mean()

        if self.mode == "fixed_var":
            vars_ = torch.full_like(diff, self.fixed_var_value)
        else:
            vars_ = self.vars[labels].clamp_min(self.var_floor)

        nll = 0.5 * (diff * diff / vars_).mean(dim=1) + 0.5 * torch.log(vars_).mean(dim=1)
        return nll.mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    Supports:
      - crossentropy (CB-CE)
      - focal (CB-Focal)
    """

    def __init__(
        self,
        samples_per_cls,
        no_of_classes,
        beta=0.9999,
        loss_type="crossentropy",
        focal_gamma=2.0,
        eps=1e-12,
    ):
        super(ClassBalancedLoss, self).__init__()
        self.loss_type = str(loss_type).lower()
        if self.loss_type in ("ce", "cb_ce"):
            self.loss_type = "crossentropy"
        elif self.loss_type in ("cb_focal",):
            self.loss_type = "focal"
        if self.loss_type not in ("crossentropy", "focal"):
            raise ValueError("loss_type must be one of: crossentropy | focal")

        self.no_of_classes = int(no_of_classes)
        self.beta = float(beta)
        self.focal_gamma = float(focal_gamma)
        self.eps = float(eps)

        counts = torch.as_tensor(samples_per_cls, dtype=torch.float64)
        if counts.numel() != self.no_of_classes:
            raise ValueError(
                f"samples_per_cls length mismatch: {counts.numel()} != {self.no_of_classes}"
            )

        valid = counts > 0
        effective_num = 1.0 - torch.pow(torch.tensor(self.beta, dtype=torch.float64), counts)
        weights = torch.zeros_like(counts, dtype=torch.float64)
        weights[valid] = (1.0 - self.beta) / torch.clamp(effective_num[valid], min=self.eps)
        if torch.sum(weights) <= 0:
            weights = torch.ones_like(weights, dtype=torch.float64)

        # Normalize so the average class weight is approximately 1.
        weights = weights / torch.clamp(weights.sum(), min=self.eps) * float(self.no_of_classes)
        self.register_buffer("weights", weights.to(torch.float32))

    def forward(self, logits, labels):
        if self.loss_type == "crossentropy":
            return F.cross_entropy(logits, labels, weight=self.weights)

        # CB-Focal: alpha from class-balanced weights and p_t from logits.
        ce = F.cross_entropy(logits, labels, reduction="none")
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, min=self.eps, max=1.0)
        alpha = self.weights.gather(0, labels)
        loss = alpha * torch.pow(1.0 - pt, self.focal_gamma) * ce
        return loss.mean()


class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()


class GCEandRS(nn.Module):
    def __init__(self, num_classes=10, q=0.7, tau=10, p=0.1, lamb=1.2):
        super(GCEandRS, self).__init__()
        self.criterion = GCELoss(num_classes=num_classes, q=q)
        self.tau = tau
        self.p = p
        self.lamb = lamb
        self.norm = pNorm(p=p)

    def forward(self, out, y):
        out = F.normalize(out, dim=1)
        loss = self.criterion(out / self.tau, y) + self.lamb * self.norm(out / self.tau, self.p)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
