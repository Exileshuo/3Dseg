import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 基础组件：深度可分离卷积 ---
class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                                   padding=1, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.pointwise(self.depthwise(x))))


# --- 核心改进：注意力门 ---
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, kernel_size=1, bias=True), nn.InstanceNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, kernel_size=1, bias=True), nn.InstanceNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, kernel_size=1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv3d(in_channels, out_channels),
            DepthwiseSeparableConv3d(out_channels, out_channels)
        )

    def forward(self, x): return self.double_conv(x)


# --- 模型主体：Attention Lite 3D U-Net ---
class Lite3DUNet_Attn(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=16):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_filters * 4, base_filters * 8))
        self.down4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base_filters * 8, base_filters * 16))

        self.up1 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=base_filters * 8, F_l=base_filters * 8, F_int=base_filters * 4)
        self.conv_up1 = DoubleConv(base_filters * 16, base_filters * 8)

        self.up2 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2)
        self.conv_up2 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up3 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=base_filters * 2, F_l=base_filters * 2, F_int=base_filters)
        self.conv_up3 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up4 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=base_filters, F_l=base_filters, F_int=base_filters // 2)
        self.conv_up4 = DoubleConv(base_filters * 2, base_filters)

        self.outc = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def pad_to_match(self, tensor, target):
        diff = [target.size()[i] - tensor.size()[i] for i in range(2, 5)]
        return F.pad(tensor, [diff[2] // 2, diff[2] - diff[2] // 2, diff[1] // 2, diff[1] - diff[1] // 2, diff[0] // 2,
                              diff[0] - diff[0] // 2])

    def forward(self, x, return_features=False):
        x1 = self.inc(x);
        x2 = self.down1(x1);
        x3 = self.down2(x2);
        x4 = self.down3(x3);
        x5 = self.down4(x4)

        u1 = self.pad_to_match(self.up1(x5), x4)
        x4 = self.att1(g=u1, x=x4)
        u1 = self.conv_up1(torch.cat([x4, u1], dim=1))

        u2 = self.pad_to_match(self.up2(u1), x3)
        x3 = self.att2(g=u2, x=x3)
        u2 = self.conv_up2(torch.cat([x3, u2], dim=1))

        u3 = self.pad_to_match(self.up3(u2), x2)
        x2 = self.att3(g=u3, x=x2)
        u3 = self.conv_up3(torch.cat([x2, u3], dim=1))

        u4 = self.pad_to_match(self.up4(u3), x1)
        x1 = self.att4(g=u4, x=x1)
        u4 = self.conv_up4(torch.cat([x1, u4], dim=1))

        logits = self.outc(u4)
        return (logits, x5) if return_features else logits


# --- 损失函数：针对 BraTS 难易程度加权 ---
class WeightedDiceFocalLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 2.0], smooth=1e-5):
        super().__init__()
        self.weights = weights
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        total_loss = 0
        for i in range(len(self.weights)):
            # Dice
            intersect = (probs[:, i] * targets[:, i]).sum(dim=(1, 2, 3))
            union = probs[:, i].sum(dim=(1, 2, 3)) + targets[:, i].sum(dim=(1, 2, 3))
            dice_loss = 1 - (2. * intersect + self.smooth) / (union + self.smooth)
            # Focal
            bce = F.binary_cross_entropy_with_logits(logits[:, i], targets[:, i], reduction='none')
            pt = torch.exp(-bce)
            focal_loss = ((1 - pt) ** 2 * bce).mean()
            total_loss += self.weights[i] * (dice_loss.mean() + focal_loss)
        return total_loss / len(self.weights)