import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, List
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
import cv2


class EnhancedMosaicGenerator:
    def __init__(self, tile_size: int = 32):
        self.tile_size = tile_size
        self.tiles = []
        self.tile_colors = []

    def load_tiles(self, tiles_folder: str):
        """Load all tile images from the specified folder."""
        self.tiles = []
        self.tile_colors = []

        for filename in os.listdir(tiles_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                tile_path = os.path.join(tiles_folder, filename)
                try:
                    tile = Image.open(tile_path)
                    tile = tile.resize((self.tile_size, self.tile_size))
                    tile = tile.convert('RGB')

                    self.tiles.append(np.array(tile))

                    # Calculate average color for matching
                    avg_color = np.mean(tile, axis=(0, 1))
                    self.tile_colors.append(avg_color)
                except Exception as e:
                    print(f"Error loading tile {filename}: {e}")

    def apply_color_quantization(self, image_array: np.ndarray,
                                n_colors: int = 16) -> np.ndarray:
        """Apply color quantization to reduce color variations."""
        if n_colors <= 0:
            return image_array

        # Reshape image to 2D array of pixels
        h, w, c = image_array.shape
        pixels = image_array.reshape(-1, c)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Replace pixels with cluster centers
        quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
        quantized_image = quantized_pixels.reshape(h, w, c).astype(np.uint8)

        return quantized_image

    def adaptive_grid_segmentation(self, image_array: np.ndarray,
                                  base_size: int = 32) -> List[Tuple]:
        """
        Implement adaptive grid segmentation starting with base_size
        and subdividing based on variance threshold.
        """
        h, w = image_array.shape[:2]
        segments = []

        def should_subdivide(region, threshold=1000):
            """Determine if a region should be subdivided based on variance."""
            if region.shape[0] < 8 or region.shape[1] < 8:
                return False
            variance = np.var(region)
            return variance > threshold

        def process_region(start_row, end_row, start_col, end_col, size):
            """Recursively process regions."""
            region = image_array[start_row:end_row, start_col:end_col]

            if should_subdivide(region) and size > 8:
                # Subdivide into 4 quadrants
                mid_row = start_row + (end_row - start_row) // 2
                mid_col = start_col + (end_col - start_col) // 2
                new_size = size // 2

                process_region(start_row, mid_row, start_col, mid_col, new_size)
                process_region(start_row, mid_row, mid_col, end_col, new_size)
                process_region(mid_row, end_row, start_col, mid_col, new_size)
                process_region(mid_row, end_row, mid_col, end_col, new_size)
            else:
                # Add this region as a final segment
                avg_color = np.mean(region, axis=(0, 1))
                segments.append((start_row, end_row, start_col, end_col,
                               avg_color, size))

        # Start with base grid
        rows = range(0, h, base_size)
        cols = range(0, w, base_size)

        for start_row in rows:
            end_row = min(start_row + base_size, h)
            for start_col in cols:
                end_col = min(start_col + base_size, w)
                process_region(start_row, end_row, start_col, end_col, base_size)

        return segments

    def find_best_tile(self, target_color: np.ndarray) -> np.ndarray:
        """Find the tile with the closest average color to the target."""
        if not self.tile_colors:
            return np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)

        distances = [np.linalg.norm(target_color - tile_color)
                     for tile_color in self.tile_colors]
        best_tile_idx = np.argmin(distances)
        return self.tiles[best_tile_idx]

    def create_colored_tile(self, color: np.ndarray, size: int) -> np.ndarray:
        """Create a solid colored tile."""
        return np.full((size, size, 3), color.astype(np.uint8), dtype=np.uint8)

    def calculate_mse(self, original: np.ndarray,
                     mosaic: np.ndarray) -> float:
        """Calculate Mean Squared Error between original and mosaic."""
        # Resize images to same size if needed
        if original.shape != mosaic.shape:
            mosaic = cv2.resize(mosaic, (original.shape[1], original.shape[0]))

        mse = np.mean((original.astype(float) - mosaic.astype(float)) ** 2)
        return mse

    def calculate_ssim(self, original: np.ndarray,
                      mosaic: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Resize images to same size if needed
        if original.shape != mosaic.shape:
            mosaic = cv2.resize(mosaic, (original.shape[1], original.shape[0]))

        # Convert to grayscale for SSIM calculation
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_RGB2GRAY)

        ssim_score = ssim(original_gray, mosaic_gray, data_range=255)
        return ssim_score

    def create_mosaic(self, input_image_path: str, output_path: str,
                     grid_size: Optional[Tuple[int, int]] = None,
                     use_adaptive_grid: bool = False,
                     use_colored_tiles: bool = False,
                     color_quantization: int = 0,
                     base_grid_size: int = 32) -> Tuple[str, dict]:
        """
        Create a mosaic from the input image with enhanced features.

        Returns:
            Tuple of (output_path, performance_metrics)
        """
        # Load and process input image
        input_image = Image.open(input_image_path)
        input_image = input_image.convert('RGB')
        original_array = np.array(input_image)

        # Apply color quantization if requested
        if color_quantization > 0:
            processed_array = self.apply_color_quantization(
                original_array, color_quantization)
        else:
            processed_array = original_array

        # Create mosaic based on method
        if use_adaptive_grid:
            mosaic = self._create_adaptive_mosaic(
                processed_array, use_colored_tiles, base_grid_size)
        else:
            mosaic = self._create_regular_mosaic(
                processed_array, grid_size, use_colored_tiles)

        # Calculate performance metrics
        mse = self.calculate_mse(original_array, mosaic)
        ssim_score = self.calculate_ssim(original_array, mosaic)

        performance_metrics = {
            'mse': mse,
            'ssim': ssim_score,
            'psnr': 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        }

        # Save mosaic
        mosaic_image = Image.fromarray(mosaic)
        mosaic_image.save(output_path)

        return output_path, performance_metrics

    def _create_adaptive_mosaic(self, image_array: np.ndarray,
                               use_colored_tiles: bool,
                               base_size: int) -> np.ndarray:
        """Create mosaic using adaptive grid segmentation."""
        segments = self.adaptive_grid_segmentation(image_array, base_size)
        h, w = image_array.shape[:2]
        mosaic = np.zeros((h, w, 3), dtype=np.uint8)

        for start_row, end_row, start_col, end_col, avg_color, size in segments:
            if use_colored_tiles:
                tile = self.create_colored_tile(avg_color, size)
            else:
                tile = self.find_best_tile(avg_color)
                if tile.shape[0] != size:
                    tile = cv2.resize(tile, (size, size))

            # Handle cases where segment size doesn't match tile size
            segment_h = end_row - start_row
            segment_w = end_col - start_col

            if tile.shape[:2] != (segment_h, segment_w):
                tile = cv2.resize(tile, (segment_w, segment_h))

            mosaic[start_row:end_row, start_col:end_col] = tile

        return mosaic

    def _create_regular_mosaic(self, image_array: np.ndarray,
                              grid_size: Optional[Tuple[int, int]],
                              use_colored_tiles: bool) -> np.ndarray:
        """Create mosaic using regular grid."""
        h, w = image_array.shape[:2]

        if grid_size is None:
            grid_height = h // self.tile_size
            grid_width = w // self.tile_size
        else:
            grid_width, grid_height = grid_size

        # Resize input image to match grid
        resized_input = cv2.resize(image_array, (grid_width, grid_height))

        # Create mosaic
        mosaic_height = grid_height * self.tile_size
        mosaic_width = grid_width * self.tile_size
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

        for row in range(grid_height):
            for col in range(grid_width):
                target_color = resized_input[row, col]

                if use_colored_tiles:
                    tile = self.create_colored_tile(target_color, self.tile_size)
                else:
                    tile = self.find_best_tile(target_color)

                start_row = row * self.tile_size
                end_row = start_row + self.tile_size
                start_col = col * self.tile_size
                end_col = start_col + self.tile_size

                mosaic[start_row:end_row, start_col:end_col] = tile

        return mosaic