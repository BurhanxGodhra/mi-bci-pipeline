import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 1001,
        F1: int = 8,          # number of temporal filters
        D: int = 2,           # depth multiplier (spatial filters per temporal filter)
        F2: int = 16,         # number of pointwise filters (F1 * D)
        kernel_length: int = 64,   # ~250ms at 250Hz, good for mu/beta rhythms
        dropout: float = 0.5,
    ):
        super().__init__()

        # --- Block 1: Temporal convolution (learn frequency-like filters) ---
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )

        # --- Block 2: Depthwise spatial convolution (learn per-channel spatial filters) ---
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # --- Block 3: Separable convolution (efficient feature combination) ---
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # --- Classifier head ---
        # Compute flattened feature size dynamically with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            feat_dim = self._forward_features(dummy).shape[1]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        x = self._forward_features(x)
        return self.classifier(x)


if __name__ == "__main__":
    # Sanity check: does a forward pass run and produce the right output shape?
    model = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
    dummy_input = torch.randn(8, 1, 22, 1001)  # batch of 8 trials
    output = model(dummy_input)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}   # should be (8, 4)")
    print(f"Trainable parameters: {n_params:,}")
