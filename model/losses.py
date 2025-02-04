import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        Weighted Focal Loss for logits.

        Args:
            alpha (float): Weight for the positive class in the range [0,1].
            gamma (float): Focusing parameter to address class imbalance.
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the Weighted Focal Loss using logits.

        Args:
            inputs (Tensor): Raw logits, shape [B, T, H, W].
            targets (Tensor): Binary ground truth, shape [B, T, H, W].

        Returns:
            Tensor: Scalar loss value.
        """
        # 1) Compute standard BCE loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 2) Convert logits to probabilities for focal factor
        p = torch.sigmoid(inputs)

        # 3) Compute p_t (true class probability)
        pt = p * targets + (1 - p) * (1 - targets)

        # 4) Gather alpha for each pixel
        alpha_t = self.alpha.to(inputs.device).gather(0, targets.long().view(-1)).view_as(targets)

        # 5) Apply focal scaling
        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce_loss

        return focal_loss.mean()


class BCEWithWeights(nn.Module):
    def __init__(self, pos_weight=None):
        """
        Binary Cross-Entropy Loss with optional positive weighting using logits.

        Args:
            pos_weight (float): Scalar weight for the positive class.
                               (For multi-class, this can be a 1D Tensor.)
        """
        super(BCEWithWeights, self).__init__()
        # Note: The built-in 'pos_weight' in BCEWithLogitsLoss is broadcasted
        #       only across the channel dimension. For binary classification,
        #       this can be a single scalar in a (1,) tensor.
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Compute BCE Loss with optional weighting using logits.

        Args:
            inputs (Tensor): Raw logits, shape [B, T, H, W].
            targets (Tensor): Binary ground truth, shape [B, T, H, W].

        Returns:
            Tensor: Scalar loss value.
        """
        if self.pos_weight is not None:
            # Option A: Use the built-in pos_weight argument
            # For a single class, pos_weight must be a tensor of shape [1]
            pos_weight_tensor = torch.tensor([self.pos_weight], device=inputs.device)
            return F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pos_weight_tensor
            )
        else:
            # Option B: No pos_weight
            return F.binary_cross_entropy_with_logits(inputs, targets)


class AsymFocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.6, gamma=0.5, smooth=1e-6):
        """
        Asymmetric Focal Tversky Loss for imbalanced segmentation.

        Args:
            delta (float): Controls the weighting of false positives and false negatives.
            gamma (float): Focal parameter controlling down-weighting of easy examples.
            smooth (float): Smoothing constant to avoid division by zero.
        """
        super(AsymFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Compute the asymmetric focal Tversky loss.

        Args:
            inputs (Tensor): Model predictions (logits), shape [B, C, H, W].
            targets (Tensor): Ground truth (binary), shape [B, C, H, W].

        Returns:
            Tensor: Loss value.
        """
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities
        tp = torch.sum(targets * inputs, dim=(2, 3))
        fn = torch.sum(targets * (1 - inputs), dim=(2, 3))
        fp = torch.sum((1 - targets) * inputs, dim=(2, 3))

        tversky_index = (tp + self.smooth) / (tp + self.delta * fn + (1 - self.delta) * fp + self.smooth)
        loss = torch.mean((1 - tversky_index) ** self.gamma)

        return loss


class AsymFocalLoss(nn.Module):
    def __init__(self, delta=0.6, gamma=2.0):
        """
        Asymmetric Focal Loss for binary segmentation.

        Args:
            delta (float): Controls weight given to false positives and false negatives.
            gamma (float): Focal parameter controlling down-weighting of easy examples.
        """
        super(AsymFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the asymmetric focal loss.

        Args:
            inputs (Tensor): Model predictions (logits), shape [B, C, H, W].
            targets (Tensor): Ground truth (binary), shape [B, C, H, W].

        Returns:
            Tensor: Loss value.
        """
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Compute the focal scaling factor
        focal_factor = torch.pow(1 - inputs, self.gamma)
        loss = self.delta * ce_loss * focal_factor * targets + (1 - self.delta) * ce_loss * (1 - targets)

        return loss.mean()


class AsymUnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        """
        Unified Asymmetric Focal Loss combining Focal Tversky Loss and Focal Loss.

        Args:
            weight (float): Lambda parameter controlling the mix of loss functions.
            delta (float): Controls weight given to false positives and false negatives.
            gamma (float): Focal parameter controlling down-weighting of easy examples.
        """
        super(AsymUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.tversky_loss = AsymFocalTverskyLoss(delta=delta, gamma=gamma)
        self.focal_loss = AsymFocalLoss(delta=delta, gamma=gamma)

    def forward(self, inputs, targets):
        """
        Compute the unified asymmetric focal loss.

        Args:
            inputs (Tensor): Model predictions (logits), shape [B, C, H, W].
            targets (Tensor): Ground truth (binary), shape [B, C, H, W].

        Returns:
            Tensor: Loss value.
        """
        loss = self.weight * self.tversky_loss(inputs, targets) + (1 - self.weight) * self.focal_loss(inputs, targets)
        return loss

