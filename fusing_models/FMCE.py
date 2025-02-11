# PyTorch Libraries
import torch.nn as nn
import torch


class FMCE(nn.Module):
    """
    Applies two sequential convolutional blocks (with dilation & skip connections)
    to the masked input, effectively fusing the masked representations.
    """

    def __init__(self, total_channels, c_sync, c_fuse, kernel_size):
        super(FMCE, self).__init__()
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

        outputs = []
        for channel_to_mask in range(x.shape[1]):
            #mask = torch.ones_like(x)
            #mask[:, channel_to_mask, :] = torch.rand_like(x[:, channel_to_mask, :])
            x_masked = x.clone()
            x_masked[:, channel_to_mask, :] = torch.rand_like(x[:, channel_to_mask, :]) 
            out_part1 = self.fusing_part1(x_masked)
            residual1 = self.residual_conv1(x_masked)
            out_part1 = out_part1 + residual1

            out_part2 = self.fusing_part2(out_part1)
            residual2 = self.residual_conv2(out_part1)
            out_part2 = out_part2 + residual2
            outputs.append(out_part2[:, channel_to_mask, :])
        
        outputs = torch.stack(outputs, dim=1)

        return outputs
