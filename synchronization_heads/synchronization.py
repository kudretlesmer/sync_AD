# Standard libraries
from itertools import accumulate

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sklearn and other utilities
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

# Custom imports
import utilities
import synchronization_heads.synchronization_utils as synchronization_utils

class SynchronizationBlock(nn.Module):
    """
    Resamples/synchronizes each sensor’s data to a common length L_common.
    Multiple methods: 'sync_head_conv', 'sync_head_fc', 'resample_linear', etc.
    """

    def __init__(
        self,
        sensors,
        num_channels,
        c_sync,
        sync_head_conv_parameters,
        params,
        synchronization_method,
        window_lengths,
        fc_num_layers
    ):
        super().__init__()  # Corrected this line
        self.default_device = params['device']

        self.L_common = sync_head_conv_parameters[
            list(sync_head_conv_parameters.keys())[0]]['input_2']

        self.total_channels = sum(num_channels.values())
        self.c_sync = c_sync
        self.synchronization_method = synchronization_method
        self.window_lengths = window_lengths

        if synchronization_method == 'sync_head_conv':
            self.sync_heads = nn.ModuleList([
                synchronization_utils.create_synchronization_head(
                    input_sensor_channels=num_channels[sensor],
                    output_sensor_channels=c_sync * num_channels[sensor],
                    groups=num_channels[sensor],
                    parameters=sync_head_conv_parameters[sensor],
                )
                for sensor in sensors
            ])
            self._sync_fn = self._resample_sync_head_conv

        elif synchronization_method == 'sync_head_fc':
            self.sync_heads = nn.ModuleList([
                synchronization_utils.create_fc_head(
                    input_size=window_lengths[sensor],      # L_in (raw)
                    output_size=self.L_common,    # L_out (common)
                    num_channels=num_channels[sensor],
                    num_layers=fc_num_layers
                )
                for sensor in sensors
            ])
            self._sync_fn = self._resample_sync_head_fc

        elif synchronization_method == 'resample_linear':
            self.sync_heads = None
            self._sync_fn = self._resample_linear

        elif synchronization_method == 'resample_fft':
            self.sync_heads = None
            self._sync_fn = self._resample_fft

        elif synchronization_method == 'zeropad':
            self.sync_heads = None
            self._sync_fn = self._resample_zeropad

        elif synchronization_method == 'resample_spline':
            self.sync_heads = None
            self._sync_fn = self._resample_spline
        else:
            raise ValueError(f"Unknown synchronization_method: {synchronization_method}")

    def forward(self, input_data_list):
        """
        input_data_list: list of (B, C_sensor, L_sensor)
        returns: (B, C_total*c_sync, L_common)
        """
        return self._sync_fn(input_data_list)

    def _resample_sync_head_conv(self, input_data_list):
        return torch.cat([
            head(inp) for head, inp in zip(self.sync_heads, input_data_list)
        ], dim=1)

    def _resample_sync_head_fc(self, input_data_list):
        return torch.cat([
            head(inp) for head, inp in zip(self.sync_heads, input_data_list)
        ], dim=1)

    def _resample_linear(self, input_data_list):
        input_data_list = [inp.cpu() for inp in input_data_list]
        resampled = [
            F.interpolate(
                inp, size=self.L_common, mode='linear'
            ).to(self.default_device)
            for inp in input_data_list
        ]
        return torch.cat(resampled, dim=1)

    def _resample_spline(self, input_data_list):
        """
        input_data_list: list of Tensors, each (B, C, T)
        self.L_common: target length for time dimension
        self.default_device: 'cpu', 'cuda', or 'mps'
        
        Returns: (B, sum_of_C, self.L_common)
        """
        resampled_list = []
        
        for inp in input_data_list:
            # Convert to float32 on CPU
            arr = inp.float().cpu().numpy()  # shape: (B, C, T)
            B, C, T = arr.shape
            
            # Original and new time points
            x = np.arange(T, dtype=np.float32)
            x_new = np.linspace(0, T - 1, self.L_common, dtype=np.float32)
            
            # Flatten to (B*C, T) so each row is one signal
            arr_flat = arr.reshape(B * C, T)
            
            spline = make_interp_spline(x, arr_flat, axis=1, k=3)
            
            # Evaluate spline => (B*C, L_common)
            arr_resampled_flat = spline(x_new)
            
            # Reshape back to (B, C, L_common), ensure float32
            arr_resampled = arr_resampled_flat.reshape(B, C, self.L_common).astype(np.float32)
            
            # Convert to torch on target device
            resampled_list.append(torch.from_numpy(arr_resampled).to(self.default_device))

        # Concatenate along channel => (B, sum_of_C, L_common)
        return torch.cat(resampled_list, dim=1)




    def _fft_resample_single(self, input_data):
        fft_vals = torch.fft.fft(input_data, dim=-1)
        L_new = self.L_common
        L = input_data.size(-1)

        if L_new > L:
            pad_size = L_new - L
            fft_vals = F.pad(fft_vals, (0, pad_size), "constant", 0)
        else:
            fft_vals = fft_vals[..., :L_new]

        return torch.fft.ifft(fft_vals, dim=-1).real

    def _resample_fft(self, input_data_list):
        input_data_list = [inp.cpu() for inp in input_data_list]
        resampled = [
            self._fft_resample_single(inp).to(self.default_device)
            for inp in input_data_list
        ]
        return torch.cat(resampled, dim=1)

    def _resample_zeropad(self, input_data_list):
        padded_data = []
        for inp in input_data_list:
            B, C, L = inp.shape
            if L < self.L_common:
                pad_size = self.L_common - L
                inp_padded = F.pad(inp, (0, pad_size), "constant", 0)
            else:
                inp_padded = inp[..., :self.L_common]
            padded_data.append(inp_padded)
        return torch.cat(padded_data, dim=1)

class DesynchronizationBlock(nn.Module):
    """
    Projects the fused representation (common length L_common)
    back to each sensor's original length L_sensor (or to any desired L_out).
    Multiple methods: 'conv' or 'fc'.
    """

    def __init__(
        self,
        sensors,
        num_channels,
        c_sync,
        params,
        sync_head_conv_parameters,
        desynchronization_method='conv',
        fc_num_layers=1,
        window_lengths=None
    ):
        super().__init__()  # Corrected this line
        self.sensors = sensors
        self.num_channels = num_channels
        self.c_sync = c_sync
        self.total_channels = sum(num_channels.values())

        channel_sizes = [num_channels[sensor] * self.c_sync for sensor in sensors]
        cumulative_offsets = [0] + list(accumulate(channel_sizes))[:-1]
        self.proj_slices = [
            slice(start, start + size)
            for start, size in zip(cumulative_offsets, channel_sizes)
        ]

        self.proj_heads = nn.ModuleList()
        if desynchronization_method == 'conv':
            for sensor in sensors:
                out_params = synchronization_utils.invert_synchronization_head_parameters(
                    sync_head_conv_parameters[sensor]
                )
                proj_head = synchronization_utils.create_synchronization_head(
                    input_sensor_channels=c_sync * num_channels[sensor],
                    output_sensor_channels=num_channels[sensor],
                    groups=num_channels[sensor],
                    parameters=out_params
                )
                self.proj_heads.append(proj_head)
            self._proj_fn = self._proj_fn_conv

        elif desynchronization_method == 'fc':
            if window_lengths is None:
                raise ValueError("For 'fc' desynchronization, you need `window_lengths`.")
            for sensor in sensors:
                L_common = sync_head_conv_parameters[sensor]['input_2']
                L_sensor = window_lengths[sensor]
                proj_head = create_fc_head(
                    input_size=L_common,
                    output_size=L_sensor,
                    num_channels=num_channels[sensor],
                    num_layers=fc_num_layers
                )
                self.proj_heads.append(proj_head)
            self._proj_fn = self._proj_fn_fc

        else:
            raise ValueError(f"Unknown desynchronization_method: {desynchronization_method}")

    def forward(self, fused_output):
        """
        fused_output: (B, C_total*c_sync, L_common)
        returns: list of (B, C_sensor, L_sensor), one for each sensor
        """
        return self._proj_fn(fused_output)

    def _proj_fn_conv(self, fused_output):
        sensor_desynchronizations = []
        for proj_head, sl in zip(self.proj_heads, self.proj_slices):
            sensor_out = proj_head(fused_output[:, sl, :])
            sensor_desynchronizations.append(sensor_out)
        return sensor_desynchronizations

    def _proj_fn_fc(self, fused_output):
        sensor_desynchronizations = []
        for proj_head, sl in zip(self.proj_heads, self.proj_slices):
            sensor_out = proj_head(fused_output[:, sl, :])
            sensor_desynchronizations.append(sensor_out)
        return sensor_desynchronizations


