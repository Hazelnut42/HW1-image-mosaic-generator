# Image Mosaic Generator

Interactive Image Mosaic Generator for assignment submission.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
gradio app.py
```

Or alternatively:
```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Features

- Variable grid sizes (8x8, 16x16, 32x32)
- Adaptive grid segmentation
- Color quantization
- Performance metrics (MSE, SSIM, PSNR)
- Custom tile upload

## Files

- `app.py` - Main application
- `enhanced_mosaic_generator.py` - Core algorithm
- `image_preprocessing.py` - Image preprocessing utilities
- `create_examples.py` - Test image generator
- `examples/` - Sample test images
- `tiles/` - Mosaic tiles