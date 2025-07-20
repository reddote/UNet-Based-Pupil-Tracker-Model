import logging
import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Model import gaze_dataset
from OldModel import efficient_b0
import os
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self):
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
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loss = []
        self.val_loss = []
        self.num_epochs = 0
        self.writer = SummaryWriter(log_dir='runs/your_experiment_name')

    def run(self):
        logger.info("Starting training process.")
        trained_model = None

        # Train the model
        trained_model = self.train_model(self.model, self.criterion, self.optimizer, self.dataloaders, num_epochs=10)

        if trained_model is not None:
            # Save the model
            torch.save(trained_model.state_dict(), 'efficientnet_b0_regression_0.pth')
            logger.info("Model saved to efficientnet_b0_regression_0.pth")
        else:
            logger.error("Trained model is None")
            exit()

    def train_model(self, model, criterion, optimizer, dataloaders, num_epochs=10):
        self.num_epochs += num_epochs
        tensorboard_counter = 0

        train_loss = 0

        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch + 1}/{num_epochs}')
            logger.info('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                counter = 0
                running_loss = 0.0
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float()  # Now labels should have shape [batch_size, 2]

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            train_loss = loss.item()
                            if counter % 10 == 0:
                                print(f"Phase: {phase} Average Loss : {loss.item()}")
                                self.writer.add_scalar('Loss/Train', train_loss, tensorboard_counter)
                        if phase == 'val':
                            val_loss = loss.item()
                            self.writer.add_scalar('Loss/Val', val_loss, tensorboard_counter)
                            print(f"Phase: {phase} Average Loss : {loss.item()}")
                            # validation_counter += 1

                        # This total loss is then added to running_loss, which accumulates the loss for the entire epoch
                        running_loss += loss.item()

                        counter += 1
                        tensorboard_counter += 1

                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                # Log the losses to TensorBoard
                # Save Loss values self.train_loss.append(epoch_loss)
                self.writer.add_scalar('Epoch_Loss/Train', epoch_loss, epoch)

                logger.info(f'{phase} Loss: {epoch_loss:.4f}')
                print(f'Epoch Loss: {epoch_loss:.4f}')
        self.writer.close()
        self.plotting()
        return model

    def plotting(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_epochs + 1), self.train_loss, label='Train Loss (MSE)')
        plt.plot(range(1, self.num_epochs + 1), self.val_loss, label='Val Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Train and Val MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main = Main()
    main.run()
