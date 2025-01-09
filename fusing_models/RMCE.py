# PyTorch Libraries
import torch.nn as nn
import torch


class RMCE(nn.Module):
    """
    Applies two sequential convolutional blocks (with dilation & skip connections)
    to the masked input, effectively fusing the masked representations.
    """

    def __init__(self, total_channels, c_sync, c_fuse, kernel_size, sensors, num_channels):
        super(RMCE, self).__init__()
        self.total_channels = total_channels
        self.c_sync = c_sync

        padding = kernel_size - 1
        self.total_middle_fuse_channels = total_channels*c_fuse
        # First fuse part
        self.fusing_part1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.total_channels * c_sync,
                out_channels=self.total_middle_fuse_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.total_middle_fuse_channels),
            nn.Conv1d(
                in_channels=self.total_middle_fuse_channels,
                out_channels=self.total_middle_fuse_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding * 2,
                dilation=4
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.total_middle_fuse_channels),
        )

        # Second fuse part
        self.fusing_part2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.total_middle_fuse_channels,
                out_channels=self.total_middle_fuse_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding * 4,
                dilation=8
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.total_middle_fuse_channels),
            nn.Conv1d(
                in_channels=self.total_middle_fuse_channels,
                out_channels=self.total_channels * c_sync,
                kernel_size=kernel_size,
                stride=1,
                padding=padding * 8,
                dilation=16
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.total_channels * c_sync),
        )

        # Residual layers
        self.residual_conv1 = nn.Conv1d(
            in_channels=self.total_channels * c_sync,
            out_channels=self.total_middle_fuse_channels,
            kernel_size=1,
            stride=1
        )
        self.residual_conv2 = nn.Conv1d(
            in_channels=self.total_middle_fuse_channels,
            out_channels=self.total_channels * c_sync,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        """
        x: (B, C_total*c_sync, L)

        returns: (B, C_total*c_sync, L) fused output
        """
        # generate random mask randomly masking out some channels of some sensors B, C, L
        mask = torch.rand(size=(
            x.shape[0], x.shape[1], x.shape[2]), dtype=torch.float32).to(x.device) > 0.95
        x = x * mask

        out_part1 = self.fusing_part1(x)
        residual1 = self.residual_conv1(x)
        out_part1 = out_part1 + residual1

        out_part2 = self.fusing_part2(out_part1)
        residual2 = self.residual_conv2(out_part1)
        out_part2 = out_part2 + residual2

        return out_part2
