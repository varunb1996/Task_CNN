import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Union, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for digits 0-9

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third conv block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x


class GasMeterReader:
    def __init__(self):
        """Initialize the GasMeterReader with necessary components."""
        logger.info("Initializing GasMeterReader...")

        # Initialize the CNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DigitCNN().to(self.device)

        # Load pre-trained weights if available
        try:
            self.model.load_state_dict(
                torch.load("digit_cnn.pth", map_location=self.device)
            )
            logger.info("Loaded pre-trained model weights")
        except:
            logger.warning("No pre-trained weights found, using untrained model")

        self.model.eval()

        # Define default ROI parameters
        self.roi = {
            "x": 0.1,  # 10% from left
            "y": 0.2,  # 20% from top
            "width": 0.8,  # 80% of image width
            "height": 0.6,  # 60% of image height
        }
        logger.info(f"ROI parameters set: {self.roi}")

    def detect_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the region of interest containing the meter display.

        Args:
            image: Input image as numpy array

        Returns:
            Cropped image containing the meter display
        """
        logger.info("Detecting ROI...")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning("No contours found, using default ROI")
            return self.extract_default_roi(image)

        # Find the largest contour that's roughly rectangular
        max_area = 0
        best_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter out tiny contours
                continue

            # Get rotated rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Check if the rectangle is roughly horizontal
            width = rect[1][0]
            height = rect[1][1]
            if width > height and width / height > 2:  # Should be wider than tall
                if area > max_area:
                    max_area = area
                    best_contour = box

        if best_contour is None:
            logger.warning("No suitable contour found, using default ROI")
            return self.extract_default_roi(image)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(best_contour)

        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Extract and return the region
        region = image[y : y + h, x : x + w]
        logger.info(f"Found meter display region with dimensions: {region.shape}")
        return region

    def extract_default_roi(self, image: np.ndarray) -> np.ndarray:
        """Extract ROI using default parameters."""
        h, w = image.shape[:2]
        x = int(w * self.roi["x"])
        y = int(h * self.roi["y"])
        width = int(w * self.roi["width"])
        height = int(h * self.roi["height"])
        return image[y : y + height, x : x + width]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CNN input.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        logger.info("Preprocessing image...")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Save debug image
        cv2.imwrite("debug_processed.jpg", cleaned)
        logger.info("Saved debug image")

        return cleaned

    def segment_digits(self, image: np.ndarray) -> list:
        """
        Segment individual digits from the meter display.

        Args:
            image: Preprocessed image as numpy array

        Returns:
            List of segmented digit images
        """
        logger.info("Segmenting digits...")

        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and sort contours from left to right
        digit_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if (
                w * h > 100 and 0.2 < aspect_ratio < 1.0
            ):  # Filter based on size and aspect ratio
                digit_contours.append((x, contour))

        digit_contours.sort(key=lambda x: x[0])

        # Extract digit images
        digits = []
        for _, contour in digit_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Add padding
            padding = 4
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            digit = image[y : y + h, x : x + w]

            # Resize to 32x32
            digit = cv2.resize(digit, (32, 32))

            # Normalize
            digit = digit.astype(np.float32) / 255.0

            digits.append(digit)

        return digits

    def predict_digit(self, digit_image: np.ndarray) -> Tuple[int, float]:
        """
        Predict a single digit using the CNN.

        Args:
            digit_image: Preprocessed digit image

        Returns:
            Tuple of (predicted digit, confidence)
        """
        # Convert to tensor
        x = torch.from_numpy(digit_image).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        return prediction.item(), confidence.item()

    def read_digits(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Read digits from the meter display using CNN.

        Args:
            image: Preprocessed image

        Returns:
            Tuple of (cleaned text, confidence score)
        """
        try:
            logger.info("Reading digits...")

            # Segment digits
            digit_images = self.segment_digits(image)

            if not digit_images:
                logger.warning("No digits found")
                return None, 0.0

            # Predict each digit
            predictions = []
            confidences = []

            for digit in digit_images:
                pred, conf = self.predict_digit(digit)
                predictions.append(str(pred))
                confidences.append(conf)

            # Combine predictions
            text = "".join(predictions)
            confidence = np.mean(confidences)

            logger.info(f"Predicted text: {text}, Confidence: {confidence}")
            return text, confidence

        except Exception as e:
            logger.error(f"Error reading digits: {str(e)}")
            return None, 0.0

    def validate_reading(self, value: float) -> bool:
        """
        Validate if the reading makes sense.

        Args:
            value: The read value

        Returns:
            Boolean indicating if the value is valid
        """
        logger.info(f"Validating reading: {value}")

        # Check if value is within reasonable range
        if value < 0 or value > 999999.999:
            logger.warning(f"Value {value} is outside expected range")
            return False

        # Check decimal places
        decimal_str = str(value).split(".")[-1]
        if len(decimal_str) > 3:
            logger.warning(f"Value {value} has too many decimal places")
            return False

        return True


def read_consumption(image_path: Union[str, Path]) -> float:
    """
    Main function to read gas meter consumption from an image.

    Args:
        image_path: Path to the image file

    Returns:
        Float value of the consumption reading
    """
    try:
        logger.info(f"Processing image: {image_path}")

        # Initialize reader
        reader = GasMeterReader()

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image at path: {image_path}")
            raise ValueError("Could not read image")
        logger.info(f"Successfully read image with shape: {image.shape}")

        # Detect and extract ROI
        roi = reader.detect_roi(image)

        # Preprocess image
        processed = reader.preprocess_image(roi)

        # Read digits
        text, confidence = reader.read_digits(processed)

        if text is None:
            logger.error("No text detected in image")
            raise ValueError("Could not reliably read the meter value")

        # Require minimum confidence
        if confidence < 0.7:
            logger.warning(
                f"Low confidence ({confidence}) but proceeding with text: {text}"
            )

        # Convert to float
        try:
            value = float(text)
        except ValueError as e:
            logger.error(f"Failed to convert prediction '{text}' to float: {e}")
            raise ValueError(f"Prediction could not be converted to float: {text}")

        # Validate reading
        if not reader.validate_reading(value):
            logger.error(f"Read value {value} is not valid for image: {image_path}")
            raise ValueError("Invalid meter reading")

        logger.info(f"Successfully read value: {value}")
        return value

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise ValueError(f"Could not read meter value: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Test the function with a sample image
    try:
        image_path = "Counter3/02 Counter3.jpg"
        value = read_consumption(image_path)
        print(f"Read value: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")
