
import torch
import torch.nn as nn

def _try_import_torchvision(model_name: str, pretrained: bool, num_classes: int):
    try:
        import torchvision
        from packaging import version
        tv = torchvision
        # handle weights API differences across versions
        if model_name == "resnet18":
            try:
                weights = tv.models.ResNet18_Weights.DEFAULT if pretrained else None
                m = tv.models.resnet18(weights=weights)
            except Exception:
                m = tv.models.resnet18(pretrained=pretrained)
            in_features = m.fc.in_features
            m.fc = nn.Linear(in_features, num_classes)
            return m
        elif model_name.startswith("vit_"):
            # torchvision ViT (0.14+)
            try:
                if model_name == "vit_b_16":
                    weights = tv.models.ViT_B_16_Weights.DEFAULT if pretrained else None
                    m = tv.models.vit_b_16(weights=weights)
                elif model_name == "vit_b_32":
                    weights = tv.models.ViT_B_32_Weights.DEFAULT if pretrained else None
                    m = tv.models.vit_b_32(weights=weights)
                else:
                    raise ValueError("Unknown ViT variant")
                in_features = m.heads.head.in_features
                m.heads.head = nn.Linear(in_features, num_classes)
                return m
            except Exception as e:
                raise e
        else:
            raise ValueError("Unknown model_name for torchvision: {}".format(model_name))
    except Exception as e:
        raise e

def _try_import_timm(model_name: str, pretrained: bool, num_classes: int):
    import timm
    m = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return m

def create_model(model_name: str = "resnet18", num_classes: int = 7, pretrained: bool = True):
    """Factory that tries torchvision first, then timm.

    Examples (torchvision): resnet18, vit_b_16
    Examples (timm):        vit_base_patch16_224, resnet50, convnext_tiny, ...
    """
    try:
        return _try_import_torchvision(model_name, pretrained, num_classes)
    except Exception:
        # fallback to timm
        return _try_import_timm(model_name, pretrained, num_classes)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.0):
        super().__init__()
        self.eps = eps
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = self.log_softmax(preds)
        loss = torch.gather(log_preds, 1, target.unsqueeze(1))
        loss = -loss.squeeze(1)
        if self.eps > 0.0:
            loss = (1 - self.eps) * loss - self.eps * log_preds.mean(dim=-1)
        return loss.mean()
