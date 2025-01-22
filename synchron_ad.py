import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import tqdm

from fusing_models.PMCE import PMCE
from fusing_models.FMCE import FMCE 

from synchronization_heads.synchronization import SynchronizationBlock, DesynchronizationBlock
import synchronization_heads.synchronization_utils as synchronization_utils
import utilities


class ModelBuilder:
    def __init__(self, PARAMS, sync_head_conv_parameters):
        """
        Initializes the model builder with parameters and references.

        Args:
            PARAMS (dict): Dictionary with model hyperparameters.
            sync_head_conv_parameters (dict): Additional params for sync_head_conv if used.
        """
        self.PARAMS = PARAMS
        self.sync_head_conv_parameters = sync_head_conv_parameters

    def build_synchronization_block(self):
        """
        Builds and returns a synchronization block based on PARAMS.
        """
        return SynchronizationBlock(
            params=self.PARAMS,
            sensors=self.PARAMS["sensors"],
            window_lengths=self.PARAMS["WINDOW_LENGTHS"],
            num_channels=self.PARAMS["NUM_CHANNELS"],
            c_sync=self.PARAMS['C_sync'],
            synchronization_method=self.PARAMS['synchronization_method'],
            fc_num_layers=self.PARAMS['fc_num_layers'],
            sync_head_conv_parameters=self.sync_head_conv_parameters
        )

    def build_desynchronization_block(self):
        """
        Builds and returns a desynchronization block based on PARAMS.
        """
        return DesynchronizationBlock(
            params=self.PARAMS,
            sensors=self.PARAMS["sensors"],
            num_channels=self.PARAMS["NUM_CHANNELS"],
            window_lengths=self.PARAMS["WINDOW_LENGTHS"],
            c_sync=self.PARAMS['C_sync'],
            desynchronization_method=self.PARAMS['desynchronization_method'],
            fc_num_layers=self.PARAMS['fc_num_layers'],
            sync_head_conv_parameters=self.sync_head_conv_parameters,
        )

    def build_fusing_block(self):
        """
        Builds and returns a fusing block based on PARAMS.
        """
        if self.PARAMS['model'] == "PMCE":
            return PMCE(
                sensors=self.PARAMS["sensors"],
                num_channels=self.PARAMS["NUM_CHANNELS"],
                c_sync=self.PARAMS['C_sync'],
                c_fuse=self.PARAMS['C_fuse'],
                kernel_size=self.PARAMS['kernel_size']
            )
        elif self.PARAMS['model'] == "FMCE":
            return FMCE(
                total_channels=sum(self.PARAMS["NUM_CHANNELS"].values()),
                c_sync=self.PARAMS['C_sync'],
                c_fuse=self.PARAMS['C_fuse'],
                kernel_size=self.PARAMS['kernel_size']
            )

    def build_all_blocks(self):
        """
        Builds and returns all three blocks as a tuple.
        """
        sync_block = self.build_synchronization_block()
        desync_block = self.build_desynchronization_block()
        fusing_block = self.build_fusing_block()
        return sync_block, desync_block, fusing_block


def train_sync_and_desync_block(
    synchronisation_block,
    desynchronisation_block,
    train_data_loader,
    valid_data_loader,
    epochs=100,
    patience=4,
    lr=0.001,
    verbose=True,
    device='cpu',
    experiment_name='default'
):
    """
    Trains the given synchronization and desynchronization blocks 
    using the provided data loaders.

    Args:
        synchronisation_block (nn.Module): The sync block to be trained.
        desynchronisation_block (nn.Module): The desync block to be trained.
        train_data_loader (DataLoader): Dataloader for training data.
        valid_data_loader (DataLoader): Dataloader for validation data.
        epochs (int): Number of training epochs.
        patience (int): Early stopping patience (in epochs).
        lr (float): Learning rate.
        verbose (bool): Prints progress messages if True.
        device (str): Device to place the model and data on. 
                      e.g., 'cuda' or 'cpu'.

    Returns:
        (list, list): (train_losses_epoch, valid_losses_epoch) 
    """
    optimizer = torch.optim.Adam(
        list(synchronisation_block.parameters())
        + list(desynchronisation_block.parameters()), lr=lr
    )

    # Move modules to device
    synchronisation_block.to(device)
    desynchronisation_block.to(device)

    best_loss = float('inf')
    non_improving_count = 0
    best_model_state = None

    train_losses_epoch = []
    valid_losses_epoch = []

    # Main loop
    for epoch in range(epochs):
        if verbose:
            print(f"\n----------- Epoch {epoch + 1} -----------")

        # --- Training ---
        synchronisation_block.train()
        desynchronisation_block.train()

        total_train_loss = 0.0
        train_loader_tqdm = tqdm.tqdm(
            train_data_loader, desc=f"Train Epoch {epoch+1}", leave=True)

        for batch_idx, x_batch in enumerate(train_loader_tqdm):
            x_batch = [x.to(device) for x in x_batch]

            optimizer.zero_grad()

            synced_output = synchronisation_block(x_batch)
            desynced_output = desynchronisation_block(synced_output)

            # Calculate MSE across all sensors
            sensor_losses = torch.cat([
                ((x_in - x_out) ** 2).mean(dim=(0, 2))
                for x_in, x_out in zip(x_batch, desynced_output)
            ])
            loss = sensor_losses.mean()

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            avg_train_loss = total_train_loss / (batch_idx + 1)
            train_loader_tqdm.set_postfix({'Train_loss': avg_train_loss})

        train_losses_epoch.append(avg_train_loss)

        # --- Validation ---
        synchronisation_block.eval()
        desynchronisation_block.eval()

        total_valid_loss = 0.0
        valid_loader_tqdm = tqdm.tqdm(
            valid_data_loader, desc=f"Valid Epoch {epoch+1}", leave=True)

        with torch.no_grad():
            for batch_idx, x_batch in enumerate(valid_loader_tqdm):
                x_batch = [x.to(device) for x in x_batch]

                synced_output = synchronisation_block(x_batch)
                desynced_output = desynchronisation_block(synced_output)

                sensor_losses = torch.cat([
                    ((x_in - x_out) ** 2).mean(dim=(0, 2))
                    for x_in, x_out in zip(x_batch, desynced_output)
                ])
                loss = sensor_losses.mean()

                total_valid_loss += loss.item()
                avg_val_loss = total_valid_loss / (batch_idx + 1)
                valid_loader_tqdm.set_postfix({'Valid_loss': avg_val_loss})

        valid_losses_epoch.append(avg_val_loss)

        # --- Early Stopping ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            non_improving_count = 0
            # Save best state
            best_model_state = {
                'synchronisation_block': synchronisation_block.state_dict(),
                'desynchronisation_block': desynchronisation_block.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss
            }
            torch.save(synchronisation_block.state_dict(),
                           f"experiments/{experiment_name}/synchronisation_block.pth")
            torch.save(synchronisation_block.state_dict(),
                           f"experiments/{experiment_name}/desynchronisation_block.pth")

        else:
            non_improving_count += 1
            if non_improving_count > patience:
                if verbose:
                    print("Stopping early due to no improvement in validation loss.")
                break

    # Load best model if found
    if best_model_state is not None:
        synchronisation_block.load_state_dict(
            best_model_state['synchronisation_block'])
        desynchronisation_block.load_state_dict(
            best_model_state['desynchronisation_block'])
        if verbose:
            print(
                f"Loaded best model from epoch {best_model_state['epoch']} "
                f"with val_loss = {best_model_state['best_loss']:.4f}"
            )

    return train_losses_epoch, valid_losses_epoch


class FusionTrainer:
    def __init__(
        self,
        synchronisation_block,
        fusing_block,
        desynchronisation_block,
        device,
        lr=1e-4,
        epochs=100,
        train_sync=False,
        patience=10
    ):
        """
        Manages training/validation of the fusing block,
        optionally also trains the synchronization & desynchronization blocks.

        Args:
            synchronisation_block (nn.Module): Synchronization block.
            fusing_block (nn.Module): Fusion block.
            desynchronisation_block (nn.Module): Desynchronization block.
            device (torch.device): Device (e.g., 'cuda' or 'cpu').
            lr (float): Learning rate.
            epochs (int): Number of epochs.
            train_sync (bool): Whether to train the synchronization & desync blocks.
            patience (int): Number of epochs to wait for improvement in val loss 
                            before early stopping.
        """
        self.synchronisation_block = synchronisation_block
        self.fusing_block = fusing_block
        self.desynchronisation_block = desynchronisation_block

        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.train_sync = train_sync
        self.patience = patience

        # Freeze/unfreeze sync + desync based on train_sync
        # (If train_sync = False, these blocks get frozen.)
        utilities.freeze_module_parameters(
            self.synchronisation_block, freeze=not self.train_sync)
        utilities.freeze_module_parameters(
            self.desynchronisation_block, freeze=not self.train_sync)

        # Move models to device
        self.synchronisation_block.to(self.device)
        self.fusing_block.to(self.device)
        self.desynchronisation_block.to(self.device)

        # Prepare optimizer parameters
        # Always optimize the fusing block
        params_to_optimize = list(self.fusing_block.parameters())
        # If training sync, also optimize sync + desync
        if self.train_sync:
            params_to_optimize += list(self.synchronisation_block.parameters())
            params_to_optimize += list(self.desynchronisation_block.parameters())

        # Define optimizer
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)

        # For logging
        self.train_losses_epoch = []
        self.valid_losses_epoch = []

    def fit(self, train_data_loader, valid_data_loader, experiment_name):
        """
        Trains the fusing block (and optionally the sync + desync blocks), end-to-end.
        Implements early stopping and reloads the best model state at the end.
        """
        best_val_loss = float("inf")
        no_improvement_count = 0
        best_model_state = None

        for epoch in range(self.epochs):
            # -----------------------
            # TRAINING PHASE
            # -----------------------
            total_train_loss = 0.0
            train_loader_tqdm = tqdm.tqdm(
                train_data_loader,
                desc=f"Train Epoch {epoch+1}",
                leave=True
            )

            # Set modes:
            self.fusing_block.train()  # Always training the fusing block
            if self.train_sync:
                # If we are training sync + desync:
                self.synchronisation_block.train()
                self.desynchronisation_block.train()
            else:
                # Otherwise, keep them in eval mode
                self.synchronisation_block.eval()
                self.desynchronisation_block.eval()

            for batch_idx, x_batch in enumerate(train_loader_tqdm):
                x_batch = [x.to(self.device) for x in x_batch]

                self.optimizer.zero_grad()

                # 1) Synchronize
                # list-of-tensors in => typically list or combined out
                synched = self.synchronisation_block(x_batch)
                # 2) Fuse
                fused = self.fusing_block(synched)

                if self.train_sync:
                    # If training sync/desync: reconstruct back and compare to original
                    x_batch_output = self.desynchronisation_block(fused)
                    # Compute reconstruction loss across all sensors
                    # (shape: (batch, channels, length))
                    sensor_losses = torch.cat([
                        ((x_sensor - x_sensor_out) ** 2).mean(dim=(0, 2))
                        for x_sensor, x_sensor_out in zip(x_batch, x_batch_output)
                    ])
                else:
                    # Not training sync/desync => do a direct MSE(synched, fused)
                    # (this only updates the fusing block unless blocks are not actually frozen)
                    sensor_losses = ((synched - fused) ** 2).mean(dim=(0, 2))

                loss = sensor_losses.mean()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                avg_train_loss = total_train_loss / (batch_idx + 1)
                train_loader_tqdm.set_postfix({'Train_loss': avg_train_loss})

            self.train_losses_epoch.append(avg_train_loss)

            # -----------------------
            # VALIDATION PHASE
            # -----------------------
            total_valid_loss = 0.0
            valid_loader_tqdm = tqdm.tqdm(
                valid_data_loader,
                desc=f"Valid Epoch {epoch+1}",
                leave=True
            )

            # Eval mode for validation
            self.fusing_block.eval()
            self.synchronisation_block.eval()
            self.desynchronisation_block.eval()

            with torch.no_grad():
                for batch_idx, x_batch in enumerate(valid_loader_tqdm):
                    x_batch = [x.to(self.device) for x in x_batch]

                    synched = self.synchronisation_block(x_batch)
                    fused = self.fusing_block(synched)

                    if self.train_sync:
                        # Reconstruct
                        x_batch_output = self.desynchronisation_block(fused)
                        sensor_losses = torch.cat([
                            ((x_sensor - x_sensor_out) ** 2).mean(dim=(0, 2))
                            for x_sensor, x_sensor_out in zip(x_batch, x_batch_output)
                        ])
                    else:
                        # Compare synched vs fused
                        sensor_losses = ((synched - fused) **
                                         2).mean(dim=(0, 2))

                    loss = sensor_losses.mean()

                    total_valid_loss += loss.item()
                    avg_val_loss = total_valid_loss / (batch_idx + 1)
                    valid_loader_tqdm.set_postfix({'Valid_loss': avg_val_loss})

            self.valid_losses_epoch.append(avg_val_loss)

            # -----------------------
            # EARLY STOPPING LOGIC
            # -----------------------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0
                # Save best model weights
                best_model_state = {
                    "fusing_block": copy.deepcopy(self.fusing_block.state_dict())
                }
                # save also to the disk
                torch.save(self.fusing_block.state_dict(),
                           f"experiments/{experiment_name}/fusing_block.pth")

                if self.train_sync:
                    best_model_state["synchronisation_block"] = copy.deepcopy(
                        self.synchronisation_block.state_dict())
                    best_model_state["desynchronisation_block"] = copy.deepcopy(
                        self.desynchronisation_block.state_dict())

                    # save also to the disk
                    torch.save(self.synchronisation_block.state_dict(
                    ), f"experiments/{experiment_name}/synchronisation_block.pth")
                    torch.save(self.desynchronisation_block.state_dict(
                    ), f"experiments/{experiment_name}/desynchronisation_block.pth")

            else:
                no_improvement_count += 1

            if no_improvement_count >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # -----------------------
        # RELOAD BEST MODEL STATE
        # -----------------------
        if best_model_state is not None:
            self.fusing_block.load_state_dict(best_model_state["fusing_block"])
            if self.train_sync:
                self.synchronisation_block.load_state_dict(
                    best_model_state["synchronisation_block"])
                self.desynchronisation_block.load_state_dict(
                    best_model_state["desynchronisation_block"])

        return self.train_losses_epoch, self.valid_losses_epoch

    def predict(self, data_loader):
        """
        Generates predictions (synched and fused outputs) on new data.

        Returns:
            all_synched (torch.Tensor): Concatenated synchronized outputs.
            all_fused   (torch.Tensor): Concatenated fused outputs.
        """
        self.fusing_block.eval()
        self.synchronisation_block.eval()
        # if you need desync in the prediction pipeline, handle similarly
        self.desynchronisation_block.eval()

        if self.train_sync:
            all_input = []
            all_output = []

        else:
            all_synched = []
            all_fused = []

        with torch.no_grad():
            for x_batch in tqdm.tqdm(data_loader, desc="Predicting", leave=True):
                x_batch = [x.to(self.device) for x in x_batch]
                synched = self.synchronisation_block(x_batch)
                fused = self.fusing_block(synched)

                if self.train_sync:
                    output = self.desynchronisation_block(fused)
                    for i in range(len(output)):
                        if len(all_input) <= i:
                            all_input.append([])
                            all_output.append([])
                        all_input[i].append(x_batch[i])
                        all_output[i].append(output[i])
                else:
                    all_synched.append(synched)
                    all_fused.append(fused)

            if self.train_sync:
                # Concatenate results across batches
                for i in range(len(all_input)):
                    all_input[i] = torch.cat(all_input[i], dim=0)
                    all_output[i] = torch.cat(all_output[i], dim=0)
                return all_input, all_output
            else:
                # Concatenate results across batches
                all_synched = torch.cat(all_synched, dim=0)
                all_fused = torch.cat(all_fused, dim=0)
                return all_synched, all_fused


def train_model(PARAMS, train_data_loader, valid_data_loader, test_data_loader, Y_test, NUM_CHANNELS, WINDOW_LENGTHS):
    if PARAMS['prefix'] != None:
        Experiment_name = f"{PARAMS['prefix']}-{PARAMS['machine']}-{PARAMS['synchronization_method']}-{
            PARAMS['pre_train_sync']}-{PARAMS['post_train_sync']}-{PARAMS['lambda']}"
    else:
        Experiment_name = f"{PARAMS['machine']}-{PARAMS['synchronization_method']}-{
            PARAMS['pre_train_sync']}-{PARAMS['post_train_sync']}-{PARAMS['lambda']}"

    # create a dictionary with name  experiments/Experiment_name
    os.makedirs(f"experiments/{Experiment_name}", exist_ok=True)

    # ---------------------------------------------------------
    # 1. Initialize the data loaders and sync_head_conv parameters
    # ---------------------------------------------------------

    # train_data_loader, valid_data_loader, test_data_loader, Y_test, NUM_CHANNELS, WINDOW_LENGTHS = utilities.initialise_dataloaders(PARAMS)

    PARAMS['WINDOW_LENGTHS'] = WINDOW_LENGTHS
    PARAMS['NUM_CHANNELS'] = NUM_CHANNELS

    sync_head_conv_parameters = synchronization_utils.initialize_parameters(SENSORS=PARAMS["sensors"],
                                                                            WINDOW_LENGTHS=PARAMS['WINDOW_LENGTHS'],
                                                                            NUM_CHANNELS=PARAMS['NUM_CHANNELS'],
                                                                            lambda_=PARAMS['lambda'])
    L_common = sync_head_conv_parameters[PARAMS["sensors"][0]]['input_2']
    PARAMS["L_common"] = L_common

    # ---------------------------------------------------------
    # 2. Build models using the ModelBuilder
    # ---------------------------------------------------------
    builder = ModelBuilder(
        PARAMS,
        sync_head_conv_parameters
    )
    synchronisation_block, desynchronisation_block, fusing_block = builder.build_all_blocks()

    # ---------------------------------------------------------
    # 3. Optionally, train the sync & desync blocks
    # ---------------------------------------------------------
    if PARAMS['pre_train_sync']:
        train_losses, valid_losses = train_sync_and_desync_block(
            synchronisation_block,
            desynchronisation_block,
            train_data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
            epochs=PARAMS['epochs'],
            patience=PARAMS['patience'],
            lr=PARAMS['lr'],
            verbose=True,
            device=PARAMS['device'],
            experiment_name=Experiment_name
        )

    # ---------------------------------------------------------
    # 4. Train the fusing block using FusionTrainer
    #    (train_sync=False if you don't want to re-train sync)
    # ---------------------------------------------------------
    trainer = FusionTrainer(
        synchronisation_block=synchronisation_block,
        fusing_block=fusing_block,
        desynchronisation_block=desynchronisation_block,
        train_sync=PARAMS['post_train_sync'],
        epochs=PARAMS['epochs'],
        patience=PARAMS['patience'],
        lr=PARAMS['lr'],
        device=PARAMS['device'],
    )

    train_losses_epoch, valid_losses_epoch = trainer.fit(
        train_data_loader, valid_data_loader, Experiment_name)
    PARAMS['train_losses_epoch'] = train_losses_epoch
    PARAMS['valid_losses_epoch'] = valid_losses_epoch

    # ---------------------------------------------------------
    # 5. Predict on new data
    # ---------------------------------------------------------
    synched_outputs, fused_outputs = trainer.predict(test_data_loader)

    # Define the file path where you want to save the JSON file
    json_file_path = f'experiments/{Experiment_name}/PARAMS.json'

    # Convert all int64 values to int
    PARAMS = {key: int(value) if isinstance(value, np.integer)
              else value for key, value in PARAMS.items()}
    # Write the PARAMS dictionary to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(PARAMS, json_file, indent=4)

    print(f"PARAMS have been written to {json_file_path}")
    # Calculate the mean squared error between fused and synchronized outputs

    if PARAMS['post_train_sync']:
        error_list = []
        for sensor_idx in range(len(PARAMS["sensors"])):
            error_sensor = (
                (fused_outputs[sensor_idx] - synched_outputs[sensor_idx]) ** 2).mean(dim=2).detach().cpu().numpy()
            error_list.append(error_sensor)
        errors_mtx = np.concatenate(error_list, axis=1)

    else:
        errors_mtx = ((fused_outputs - synched_outputs) **
                      2).mean(dim=2).detach().cpu().numpy()  # N, C

    # Construct a matrix with proper naming: sensor_name + channel
    column_names = [f"{sensor}_{channel}" for sensor in PARAMS["sensors"]
                    for channel in range(NUM_CHANNELS[sensor])]

    # Create a DataFrame with the errors
    error_df = pd.DataFrame(errors_mtx, columns=column_names)

    # Concatenate the error DataFrame with the Y_test DataFrame
    Y_test_c = pd.concat([Y_test.copy(), error_df], axis=1)

    # Group by segment_id and aggregate the error columns
    Y_test_c = utilities.group_by_segment_id(Y_test_c, column_names)
    AUC_results = []
    for col in column_names:
        AUC = utilities.calculate_single_auc(Y_test_c, col)
        AUC_results.append(AUC)

    AUC_results = pd.concat(AUC_results)
    AUC_results
    AUC_results.to_csv(f"experiments/{Experiment_name}/AUC_results.csv")
