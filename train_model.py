import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import logging
from gas_meter_reader import DigitCNN, GasMeterReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GasMeterDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = list(self.data_dir.glob("**/*.jpg"))
        self.reader = GasMeterReader()

        # Create a mapping of image paths to their expected values
        # This should be replaced with actual ground truth values
        self.ground_truth = {}
        for img_path in self.image_paths:
            # For now, we'll use a placeholder value
            # In practice, you should load actual ground truth values
            self.ground_truth[img_path] = "000000"  # Placeholder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))

        # Get ground truth value for this image
        expected_value = self.ground_truth[image_path]

        # Extract ROI and preprocess
        roi = self.reader.detect_roi(image)
        processed = self.reader.preprocess_image(roi)

        # Segment digits
        digits = self.reader.segment_digits(processed)

        # Convert to tensors and create labels
        digit_tensors = []
        labels = []

        # Ensure we have the same number of digits as expected
        if len(digits) != len(expected_value):
            # Pad or truncate digits to match expected length
            if len(digits) < len(expected_value):
                # Pad with zeros
                digits.extend(
                    [np.zeros((32, 32), dtype=np.float32)]
                    * (len(expected_value) - len(digits))
                )
            else:
                # Truncate
                digits = digits[: len(expected_value)]

        for i, digit in enumerate(digits):
            if self.transform:
                digit = self.transform(digit)
            else:
                # Ensure digit is float32 before converting to tensor
                digit = digit.astype(np.float32)
                digit = torch.from_numpy(digit).float().unsqueeze(0)
            digit_tensors.append(digit)
            labels.append(int(expected_value[i]))

        return {
            "digits": torch.stack(digit_tensors),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, batch in enumerate(train_loader):
            digits = (
                batch["digits"].to(device).float()
            )  # [batch, num_digits, 1, 32, 32]
            labels = batch["labels"].to(device)  # [batch, num_digits]

            # Flatten batch and digits
            batch_size, num_digits, c, h, w = digits.shape
            digits = digits.view(-1, c, h, w)  # [(batch*num_digits), 1, 32, 32]
            labels = labels.view(-1)  # [(batch*num_digits)]

            outputs = model(digits)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

            if i % 10 == 9:
                accuracy = 100 * correct / total
                logger.info(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}, Accuracy: {accuracy:.2f}%"
                )
                running_loss = 0.0
                correct = 0
                total = 0

        torch.save(model.state_dict(), f"digit_cnn_epoch_{epoch+1}.pth")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Define transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Create dataset and dataloader
    dataset = GasMeterDataset("Counter1", transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = DigitCNN().to(device)
    # Ensure model parameters are float32
    model = model.float()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(model, train_loader, criterion, optimizer, device)

    # Save final model
    torch.save(model.state_dict(), "digit_cnn.pth")
    logger.info("Training completed and model saved")


if __name__ == "__main__":
    main()
