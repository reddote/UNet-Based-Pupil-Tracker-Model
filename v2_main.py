import logging
import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import gaze_dataset
import efficient_b0
import os
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# Setup logging
file_handler = logging.FileHandler(filename='log.txt')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('Logs')

# Define transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Main:
    def __init__(self, alpha=1.0, beta=0.5, gamma=1.0):
        # Load datasets
        self.dataset = gaze_dataset.GazeDataset(
            phase='train',
            transform=data_transforms
        )
        self.datasetVal = gaze_dataset.GazeDataset(
            phase='val',
            transform=data_transforms
        )

        # Create DataLoaders
        self.train_dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=4)
        self.val_dataloader = DataLoader(self.datasetVal, batch_size=32, shuffle=False,
                                         num_workers=4)  # No shuffle for validation

        # Include the validation dataloader in a dictionary with the training dataloader
        self.dataloaders = {
            'train': self.train_dataloader,
            'val': self.val_dataloader
        }

        # Initialize EfficientNetB0
        self.model = efficient_b0.EfficientNetB0Regression().to(device)

        # multi task criterion
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

        # Define weights for each class (higher weight for minority classes)
        class_weights = torch.tensor([10.0, 1.0])  # Pupil, Background, np_zero
        self.segmentation_loss = nn.CrossEntropyLoss()
        # criterion ends

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loss = []
        self.val_loss = []
        self.num_epochs = 0
        self.writer = SummaryWriter(log_dir='runs/your_experiment_name')
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def run(self):
        logger.info("Starting training process.")
        trained_model = None

        # Train the model
        trained_model = self.train_model(self.model, self.segmentation_loss, self.optimizer, self.dataloaders,
                                         num_epochs=5)

        if trained_model is not None:
            # Save the model
            torch.save(trained_model.state_dict(), 'efficientnet_b0_regression_0.pth')
            logger.info("Model saved to efficientnet_b0_regression_0.pth")
        else:
            logger.error("Trained model is None")
            exit()

    def multi_task_criterion(self, predictions, targets):
        # Predictions: [center_x, center_y, radians(angle), width, height]
        # Targets: [center_x, center_y, radians(angle), width, height]

        # Center loss
        loss_center = self.mse_loss(predictions[:, 0:2], targets[:, 0:2])  # center_x, center_y

        # Angle loss
        loss_angle = self.mse_loss(predictions[:, 2], targets[:, 2])  # radians(angle)

        # Size loss
        loss_size = self.smooth_l1_loss(predictions[:, 3:], targets[:, 3:])  # width, height

        # Combine losses
        total_loss = self.alpha * loss_center + self.beta * loss_angle + self.gamma * loss_size

        return total_loss

    def train_model(self, model, criterion, optimizer, dataloaders, num_epochs=10):
        self.num_epochs += num_epochs
        tensorboard_counter = 0
        validation_counter = 0

        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch + 1}/{num_epochs}')
            logger.info('-' * 10)

            model.train()  # Set model to training mode
            train_loss = 0
            running_loss = 0.0

            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device).long()  # Now labels should have shape [batch_size, 2]
                labels = labels.squeeze(1)
                outputs = model(inputs)
                # print(f"input shape : {inputs.shape}")
                # print(f"Outputs Shape: {outputs.shape}")  # Expected: [batch_size, num_classes, height, width]
                # print(f"Labels Shape: {labels.shape}")  # Expected: [batch_size, height, width]
                # print(f"Labels Type: {labels.dtype}")  # Expected: torch.int64 (long)
                # print(f"Unique Labels: {torch.unique(labels)}")  # Check valid class indices
                # print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                running_loss += train_loss
                # if tensorboard_counter % 10 == 0:
                #     print(f"Phase: Train Average Loss : {loss.item()}")
                self.writer.add_scalar('Loss/Train', train_loss, tensorboard_counter)
                tensorboard_counter += 1

            epoch_loss = running_loss / len(dataloaders['train'].dataset)
            # Log the losses to TensorBoard
            # Save Loss values self.train_loss.append(epoch_loss)
            self.writer.add_scalar('Epoch_Loss/Train', epoch_loss, epoch)
            logger.info(f'Train Loss: {epoch_loss:.4f}')

            model.eval()  # Set model to evaluate mode
            val_loss = 0
            running_loss = val_loss
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device).long()  # Now labels should have shape [batch_size, 2]
                labels = labels.squeeze(1)
                outputs = model(inputs)
                with torch.no_grad():
                    loss = criterion(outputs, labels)
                    val_loss = loss.item()
                    running_loss += val_loss
                    # if tensorboard_counter % 10 == 0:
                    #     print(f"Phase: Val Average Loss : {loss.item()}")
                    self.writer.add_scalar('Loss/Val', val_loss, validation_counter)
                    # This total loss is then added to running_loss, which accumulates the loss for the entire epoch
                    validation_counter += 1
            val_epoch_loss = running_loss / len(dataloaders['val'].dataset)
            print(f'Before LR: {self.lr_scheduler.get_last_lr()}')
            self.lr_scheduler.step(val_epoch_loss)
            # Log the losses to TensorBoard
            # Save Loss values self.train_loss.append(epoch_loss)
            self.writer.add_scalar('Epoch_Loss/Val', val_epoch_loss, epoch)
            logger.info(f'LR: {self.lr_scheduler.get_last_lr()}')
            logger.info(f'Val Loss: {val_epoch_loss:.4f}')

            print(f'Epoch Train Loss: {epoch_loss:.4f}')
            print(f'Epoch Val Loss: {val_epoch_loss:.4f}')
            print(f'After LR: {self.lr_scheduler.get_last_lr()}')
            # Save the model
            torch.save(model.state_dict(), f'{epoch}_efficientnet_b0_regression_0.pth')
            logger.info(f"Model saved to {epoch}_efficientnet_b0_regression_0.pth")
        self.writer.close()
        return model


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main = Main()
    main.run()
