# PyTorch Libraries
import torch.nn as nn
import torch


class PMCE(nn.Module):
    """
    Applies two sequential convolutional blocks (with dilation & skip connections)
    to the masked input, effectively fusing the masked representations.
    """

    def __init__(self, total_channels, c_sync, c_fuse, kernel_size, sensors, num_channels):
        super(PMCE, self).__init__()
        self.total_channels = total_channels
        self.c_sync = c_sync

        # After sync, shape => (B, C_total*c_sync, L)
        # MaskOutOneChannel => (B, (C_total-c_sync)*C_total, L)
        self.total_input_fuse_channels = (
            self.total_channels - 1
        ) * self.c_sync * self.total_channels

        self.total_middle_fuse_channels = self.total_input_fuse_channels * c_fuse

        padding = kernel_size - 1
        self.mask_module = MaskOutOneChannel(sensors, num_channels, c_sync)

        # First fuse part
        self.fusing_part1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.total_input_fuse_channels,
                out_channels=self.total_middle_fuse_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=2,
                groups=self.total_channels
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.total_middle_fuse_channels),
            nn.Conv1d(
                in_channels=self.total_middle_fuse_channels,
                out_channels=self.total_middle_fuse_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding * 2,
                dilation=4,
                groups=self.total_channels
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
                dilation=8,
                groups=self.total_channels
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.total_middle_fuse_channels),
            nn.Conv1d(
                in_channels=self.total_middle_fuse_channels,
                out_channels=self.total_channels * c_sync,
                kernel_size=kernel_size,
                stride=1,
                padding=padding * 8,
                dilation=16,
                groups=self.total_channels
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.total_channels * c_sync),
        )

        # Residual layers
        self.residual_conv1 = nn.Conv1d(
            in_channels=self.total_input_fuse_channels,
            out_channels=self.total_middle_fuse_channels,
            kernel_size=1,
            stride=1,
            groups=self.total_channels
        )
        self.residual_conv2 = nn.Conv1d(
            in_channels=self.total_middle_fuse_channels,
            out_channels=self.total_channels * c_sync,
            kernel_size=1,
            stride=1,
            groups=self.total_channels
        )

    def forward(self, x):
        """
        x: (B, C_total*c_sync, L)

        returns: (B, C_total*c_sync, L) fused output
        """
        masked_groups = self.mask_module(x)

        out_part1 = self.fusing_part1(masked_groups)
        residual1 = self.residual_conv1(masked_groups)
        out_part1 = out_part1 + residual1

        out_part2 = self.fusing_part2(out_part1)
        residual2 = self.residual_conv2(out_part1)
        out_part2 = out_part2 + residual2

        return out_part2


class MaskOutOneChannel(nn.Module):
    """
    For an input of shape (B, C_total, L), this module produces a 
    concatenation of masked versions. For each block of width c_sync,
    we omit that block and keep the rest, repeating for all channel blocks.
    """

    def __init__(self, sensors, num_channels, c_sync):
        super(MaskOutOneChannel, self).__init__()
        self.sensors = sensors
        self.num_channels_dict = num_channels
        self.c_sync = c_sync

        # total_channels = sum of sensor channels
        self.total_channels = sum(num_channels.values())
        # So total input channels = total_channels * c_sync
        self.total_c = self.total_channels * self.c_sync

        self.register_buffer('final_indices', self._create_indices())

    def _create_indices(self):
        """
        For each channel block of width c_sync, omit that block
        and keep the others. Then concatenate for all blocks.
        """
        all_indices = []
        for ch in range(self.total_channels):
            remove_start = ch * self.c_sync
            remove_end = remove_start + self.c_sync
            keep_indices = list(range(0, remove_start)) + \
                list(range(remove_end, self.total_c))
            all_indices.extend(keep_indices)
        return torch.tensor(all_indices, dtype=torch.long)

    def forward(self, x):
        """
        x: shape (B, C_total, L)
        returns: shape (B, C_total*(C_total - c_sync), L)
        """
        return x.index_select(dim=1, index=self.final_indices)
