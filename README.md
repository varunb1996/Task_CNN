# Gas Meter Reader

This project implements a solution for reading gas meter values from images using computer vision and OCR techniques.

## Features

- Robust image preprocessing to handle various lighting conditions
- Region of Interest (ROI) extraction to focus on the meter display
- OCR-based digit recognition using EasyOCR
- Validation of meter readings
- Comprehensive error handling

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. The first time you run the script, it will download the EasyOCR model automatically.

## Usage

```python
from gas_meter_reader import read_consumption_from_image

# Read a single image
try:
    value = read_consumption_from_image("path/to/image.jpg")
    print(f"Gas meter reading: {value}")
except ValueError as e:
    print(f"Error: {e}")
```

## How it Works

1. **Image Preprocessing**:

   - Converts image to grayscale
   - Applies adaptive thresholding
   - Removes noise

2. **ROI Extraction**:

   - Focuses on the meter display area
   - Adjustable ROI parameters

3. **Digit Recognition**:

   - Uses EasyOCR for robust text recognition
   - Cleans and validates the recognized digits

4. **Validation**:
   - Checks if the reading is within valid range
   - Validates decimal places
   - Ensures confidence score is sufficient

## Error Handling

The system handles various error cases:

- Unreadable images
- Invalid meter readings
- Poor image quality
- Missing or corrupted files

## Notes

- The ROI parameters may need adjustment based on your specific meter layout
- The confidence threshold (0.5) can be adjusted based on your needs
- The system is designed to handle various lighting conditions and image qualities
