from typing import Tuple


class AddGaussianNoise:
    def __init__(self, std: float = 0.01, p: float = 0.25):
        self.std = float(std)
        self.p = float(p)

    def __call__(self, tensor):
        import torch

        if torch.rand(1).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return (tensor + noise).clamp(0.0, 1.0)


def build_model(model_name: str = "efficientnetv2_b0", pretrained: bool = True):
    import torch.nn as nn
    import torchvision.models as models

    name = (model_name or "").lower()
    if name in {"efficientnetv2_b0", "tf_efficientnetv2_b0"}:
        has_v2_b0 = hasattr(models, "efficientnet_v2_b0") and hasattr(models, "EfficientNet_V2_B0_Weights")
        if name == "efficientnetv2_b0" and has_v2_b0:
            weights = models.EfficientNet_V2_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_v2_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 1)
            return model
        try:
            import timm

            return timm.create_model("tf_efficientnetv2_b0", pretrained=pretrained, num_classes=1)
        except Exception as exc:
            raise RuntimeError(
                "EfficientNetV2-B0 is not available from torchvision and timm failed. "
                "Install timm or use model_name: efficientnetb0."
            ) from exc
    if name == "efficientnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model
    raise ValueError(f"Unsupported model name: {model_name}")


def get_transforms(image_size: int = 224):
    import torchvision.transforms as T

    train_tf = T.Compose(
        [
            T.Resize((image_size + 20, image_size + 20)),
            T.RandomAffine(
                degrees=7,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=0,
                fill=0,
            ),
            T.CenterCrop((image_size, image_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.25),
            T.ToTensor(),
            AddGaussianNoise(std=0.01, p=0.2),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, eval_tf
