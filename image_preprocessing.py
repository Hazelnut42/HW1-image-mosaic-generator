import os
from PIL import Image
from typing import List, Tuple


def resize_image(image_path: str,
                 max_size: Tuple[int, int] = (800, 600)) -> Image.Image:
    # Resize image but keep the same proportions
    image = Image.open(image_path)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def create_sample_tiles(output_folder: str,
                        colors: List[Tuple[int, int, int]] = None):
    # Make some basic colored square tiles
    if colors is None:
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 192, 203),  # Pink
            (165, 42, 42),  # Brown
            (128, 128, 128),  # Gray
            (0, 0, 0),      # Black
            (255, 255, 255),  # White
        ]

    os.makedirs(output_folder, exist_ok=True)

    for i, color in enumerate(colors):
        # Make a simple colored square
        tile = Image.new('RGB', (100, 100), color)
        tile_path = os.path.join(output_folder, f'tile_{i:03d}.png')
        tile.save(tile_path)

    print(f"Created {len(colors)} sample tiles in {output_folder}")


def validate_image(image_path: str) -> bool:
    # Check if this is actually an image file
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_supported_formats() -> List[str]:
    # What image types we can handle
    return ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']


def preprocess_tiles_folder(tiles_folder: str, target_size: int = 50):
    # Make all tile images the same size (50x50)
    if not os.path.exists(tiles_folder):
        raise FileNotFoundError(f"Tiles folder not found: {tiles_folder}")

    processed_count = 0
    supported_formats = get_supported_formats()

    for filename in os.listdir(tiles_folder):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            file_path = os.path.join(tiles_folder, filename)

            try:
                with Image.open(file_path) as img:
                    # Make sure it's RGB format
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize to 50x50
                    img_resized = img.resize(
                        (target_size, target_size),
                        Image.Resampling.LANCZOS)
                    # Save it back
                    img_resized.save(file_path)
                    processed_count += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Processed {processed_count} tile images")
    return processed_count
