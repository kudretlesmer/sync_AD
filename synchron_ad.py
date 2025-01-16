import torch
import torch.nn as nn
import tqdm

from synchronization_heads.synchronization import SynchronizationBlock
from synchronization_heads.synchronization import DesynchronizationBlock
from fusing_models.PMCE import PMCE
from fusing_models.RMCE import RMCE

class SynchronMaskEstimator(nn.Module):
    """
    A neural network module that:
      1) Synchronizes input sensor data to a common window length (L_common).
      2) Passes these masked versions through a fusing block (PMCE or RMCE).
      3) Desynchronizes the fused representation back to each sensor's space (L_sensor).
    """

    def __init__(
        self,
        sensors,
        num_channels,
        window_lengths,
        c_sync,
        c_fuse,
        kernel_size,
        params,
        sync_head_conv_parameters,
        sync_method,
        desynchronization_method='conv',
        fc_num_layers=1,
        fusing_block_type='PMCE'  
        # You can add more parameters here if needed.
    ):
        super(SynchronMaskEstimator, self).__init__()
        self.sensors = sensors
        self.num_channels_dict = num_channels
        self.c_sync = c_sync
        self.c_fuse = c_fuse
        self.kernel_size = kernel_size
        self.params = params
        self.default_device = params['device']
        self.total_channels = sum(num_channels.values())

        # 1) Synchronization block
        self.synchronization_block = SynchronizationBlock(
            sensors=sensors,
            num_channels=num_channels,
            c_sync=c_sync,
            sync_head_conv_parameters=sync_head_conv_parameters,
            params=params,
            sync_method=sync_method,
            window_lengths=window_lengths,
            fc_num_layers=fc_num_layers
        )

        # 2) Fusing block (choose PMCE or RMCE, or your own architecture)
        if fusing_block_type == 'PMCE':
            self.fusing_block = PMCE(
                total_channels=self.total_channels,
                c_sync=self.c_sync,
                c_fuse=self.c_fuse,
                kernel_size=self.kernel_size,
                sensors=self.sensors,
                num_channels=self.num_channels_dict
            )
        elif fusing_block_type == 'RMCE':
            self.fusing_block = RMCE(
                total_channels=self.total_channels,
                c_sync=self.c_sync,
                c_fuse=self.c_fuse,
                kernel_size=self.kernel_size,
                sensors=self.sensors,
                num_channels=self.num_channels_dict
            )
        else:
            raise ValueError(f"Unknown fusing_block_type: {fusing_block_type}")

        # 3) Desynchronization (desynchronization) block
        self.desynchronization_block = DesynchronizationBlock(
            sensors=sensors,
            num_channels=num_channels,
            c_sync=c_sync,
            sync_head_conv_parameters=sync_head_conv_parameters,
            desynchronization_method=desynchronization_method,
            fc_num_layers=fc_num_layers,
            window_lengths=window_lengths
        )

        # Optionally store best model / losses
        self.best_model_state = None

    def forward(self, input_data_list):
        """
        input_data_list: list of (B, C_sensor, L_sensor)
        returns: list of (B, C_sensor, L_sensor)
        """
        # Step 1: Synchronize & concat => (B, C_total*c_sync, L_common)
        synced_data = self.synchronization_block(input_data_list)

        # Step 2: Fuse => (B, C_total*c_sync, L_common)
        fused_output = self.fusing_block(synced_data)

        # Step 3: Project back => list of (B, C_sensor, L_sensor)
        sensor_outputs = self.desynchronization_block(synced_data)

        return sensor_outputs

    def fit(
        self,
        train_data_loader,
        valid_data_loader,
        optimizer,
        epochs=100,
        patience=4,
        verbose=True
    ):
        """
        Trains the SynchronMaskEstimator model using the given data loaders and optimizer.

        Args:
            train_data_loader (DataLoader): Training data loader.
            valid_data_loader (DataLoader): Validation data loader.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            epochs (int): Maximum number of training epochs.
            patience (int): Early stopping patience (in epochs).
            verbose (bool): Whether to print progress messages.

        Returns:
            (list, list): Two lists containing the training and validation losses per epoch.
        """
        # Move model to the default device (e.g., 'cpu', 'cuda', 'mps')
        self.to(self.default_device)

        best_loss = float('inf')
        non_improving_count = 0
        train_losses_epoch = []
        valid_losses_epoch = []

        for epoch in range(epochs):
            if verbose:
                print(f"----------- Epoch {epoch + 1} -----------")

            # ---------------------------------------------------------
            # Training phase
            # ---------------------------------------------------------
            self.train()  # Set model to training mode
            total_train_loss = 0.0

            train_loader_tqdm = tqdm.tqdm(train_data_loader, desc=f"Train Epoch {epoch+1}", leave=False)
            for batch_idx, x_batch in enumerate(train_loader_tqdm):
                optimizer.zero_grad()
                # Move each sensor in batch to device
                x_batch = [x.to(self.default_device) for x in x_batch]

                # Forward pass
                x_batch_output = self(x_batch)

                # Compute reconstruction loss across all sensors
                sensor_losses = torch.cat([
                    ((x_sensor - x_sensor_out) ** 2).mean(dim=(0, 2))  # (N, C, L) => MSE => (C,) => ...
                    for x_sensor, x_sensor_out in zip(x_batch, x_batch_output)
                ])
                loss = sensor_losses.sum()

                # Backprop & update weights
                loss.backward()
                optimizer.step()

                # Update training loss
                total_train_loss += loss.item()
                avg_train_loss = total_train_loss / (batch_idx + 1)

                train_loader_tqdm.set_postfix({'Train_loss': avg_train_loss})
            
            train_losses_epoch.append(avg_train_loss)

            # ---------------------------------------------------------
            # Validation phase
            # ---------------------------------------------------------
            self.eval()  # Set model to eval mode
            total_valid_loss = 0.0

            with torch.no_grad():
                valid_loader_tqdm = tqdm.tqdm(valid_data_loader, desc=f"Valid Epoch {epoch+1}", leave=False)
                for batch_idx, x_batch in enumerate(valid_loader_tqdm):
                    x_batch = [x.to(self.default_device) for x in x_batch]

                    x_batch_output = self(x_batch)

                    sensor_losses = torch.cat([
                        ((x_sensor - x_sensor_out) ** 2).mean(dim=(0, 2))
                        for x_sensor, x_sensor_out in zip(x_batch, x_batch_output)
                    ])
                    loss = sensor_losses.sum()
                    total_valid_loss += loss.item()
                    avg_val_loss = total_valid_loss / (batch_idx + 1)

                    valid_loader_tqdm.set_postfix({'Valid_loss': avg_val_loss})

            valid_losses_epoch.append(avg_val_loss)

            # ---------------------------------------------------------
            # Early stopping logic
            # ---------------------------------------------------------
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                # Keep a copy of the best model's state_dict
                self.best_model_state = {
                    'model_state': self.state_dict(),
                    'epoch': epoch + 1,
                    'best_loss': best_loss
                }
                non_improving_count = 0
            else:
                non_improving_count += 1

            if non_improving_count > patience:
                if verbose:
                    print("Stopping early due to no improvement in validation loss.")
                break

        # Optionally load the best state back into the model:
        if self.best_model_state is not None:
            self.load_state_dict(self.best_model_state['model_state'])

        return train_losses_epoch, valid_losses_epoch

    def predict(self, data_loader):
        """
        A simple helper method for inference/reconstruction on a given DataLoader.
        Returns a list of input tensors and a list of output tensors (concatenated across all batches).
        """
        self.eval()
        all_inputs = [[] for _ in range(len(self.sensors))]
        all_outputs = [[] for _ in range(len(self.sensors))]

        with torch.no_grad():
            for x_batch in data_loader:
                x_batch = [x.to(self.default_device) for x in x_batch]
                out_batch = self(x_batch)

                for sensor_idx in range(len(self.sensors)):
                    all_inputs[sensor_idx].append(x_batch[sensor_idx].detach().cpu())
                    all_outputs[sensor_idx].append(out_batch[sensor_idx].detach().cpu())


        # Perform final concatenation on CPU
        all_inputs = [torch.cat(tensors, dim=0) for tensors in all_inputs]
        all_outputs = [torch.cat(tensors, dim=0) for tensors in all_outputs]
        del x_batch, out_batch

        return all_inputs, all_outputs
