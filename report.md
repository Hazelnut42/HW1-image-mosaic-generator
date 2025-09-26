# Interactive Image Mosaic Generator - Technical Report

## Abstract

This report presents an implementation of an interactive image mosaic generator that reconstructs input images using smaller tile images arranged in a grid pattern. The system features adaptive grid segmentation, color quantization, and comprehensive performance evaluation through multiple quality metrics. A user-friendly Gradio interface enables real-time parameter adjustment and visualization of results.

## 1. Methodology

### 1.1 Core Algorithm

The mosaic generation process follows a systematic approach:

1. **Image Preprocessing**: Input images are resized while maintaining aspect ratio using LANCZOS resampling for optimal quality preservation.

2. **Grid Segmentation**: The system supports three fixed grid sizes (8×8, 16×16, 32×32) and an adaptive segmentation mode that dynamically subdivides regions based on image complexity.

3. **Tile Matching**: For each grid cell, the system calculates the average RGB color and finds the best matching tile using Euclidean distance in RGB color space.

4. **Mosaic Assembly**: Selected tiles are resized and positioned to reconstruct the final mosaic image.

### 1.2 Adaptive Grid Segmentation

The adaptive segmentation algorithm recursively subdivides image regions based on color variance:

```python
def should_subdivide(region, threshold=1000):
    variance = np.var(region)
    return variance > threshold and size > 8
```

This approach ensures that complex regions with high color variation receive finer detail representation, while uniform areas maintain efficiency with larger tiles.

### 1.3 Color Quantization

The system implements K-means clustering for color quantization, reducing the color palette to create artistic effects:

- Quantization levels: 0 (disabled), 4, 8, 12, 16, 20, 24, 28, 32 colors
- Uses scikit-learn's MiniBatchKMeans for computational efficiency
- Applied to both input images and tile matching process

### 1.4 Tile Management

The system provides flexible tile management with two operational modes:

1. **Image Tiles**: Custom uploaded images automatically processed to uniform 50×50 pixel tiles with RGB conversion and LANCZOS resampling for quality preservation
2. **Colored Squares**: 13 predefined solid color tiles covering the RGB spectrum (red, green, blue, yellow, magenta, cyan, orange, purple, pink, brown, gray, black, white)

**Enhanced Tile Interface Features**:
- Multi-file upload with batch processing
- Real-time gallery preview in 4×3 grid layout
- Automatic format validation (JPG, PNG, BMP, GIF, TIFF, WebP)
- One-click reset to default colored tiles
- Dynamic tile count display and refresh functionality

## 2. Performance Metrics

### 2.1 Evaluation Criteria

Three quantitative metrics assess mosaic quality:

**Mean Squared Error (MSE)**:
```
MSE = (1/N) × Σ(I_original - I_mosaic)²
```
Measures pixel-level reconstruction accuracy. Lower values indicate better quality.

**Structural Similarity Index (SSIM)**:
```
SSIM = (2μ_xμ_y + c1)(2σ_xy + c2) / (μ_x² + μ_y² + c1)(σ_x² + σ_y² + c2)
```
Evaluates structural information preservation considering luminance, contrast, and structure. Range: 0-1, higher values indicate better quality.

**Peak Signal-to-Noise Ratio (PSNR)**:
```
PSNR = 20 × log10(MAX_I / √MSE)
```
Measures reconstruction quality in decibels. Higher values indicate better quality.

### 2.2 Performance Analysis

Testing was conducted using two different tile sets on three synthetic images:

#### 2.2.1 Colored Squares Baseline Results

**Gradient Image (Smooth Transitions)**:
- Fixed 32×32: MSE=2309.4, SSIM=0.7994, PSNR=14.50
- Adaptive grid: MSE=2336.7, SSIM=0.7922, PSNR=14.44

**Geometric Image (Sharp Boundaries)**:
- Fixed 32×32: MSE=2047.9, SSIM=0.8488, PSNR=15.02
- Adaptive grid: MSE=1461.0, SSIM=0.8301, PSNR=16.48

**High Contrast Image (Checkerboard)**:
- Fixed 32×32: MSE=7468.8, SSIM=0.6624, PSNR=9.40
- Adaptive grid: MSE=1516.9, SSIM=0.8240, PSNR=16.32

#### 2.2.2 Custom Image Tiles Results

Using 5 Vancouver landscape photographs (blue seascape, pink spring, red autumn, white winter, yellow sunset) combined with the 13 colored squares:

**Gradient Image (Smooth Transitions)**:
- Fixed 32×32: MSE=3396.2, SSIM=0.2539, PSNR=12.82
- Adaptive grid: MSE=3558.6, SSIM=0.2659, PSNR=12.62

**Geometric Image (Sharp Boundaries)**:
- Fixed 32×32: MSE=2259.9, SSIM=0.7930, PSNR=14.59
- Adaptive grid: MSE=1731.9, SSIM=0.7754, PSNR=15.75

**High Contrast Image (Checkerboard)**:
- Fixed 32×32: MSE=7317.5, SSIM=0.6598, PSNR=9.49
- Adaptive grid: MSE=1522.9, SSIM=0.8203, PSNR=16.30

## 3. Results and Discussion

### 3.1 Key Findings

1. **Tile Type Impact**: The choice between colored squares and photographic tiles significantly affects reconstruction quality. Custom image tiles show markedly different performance characteristics:
   - **Gradient images**: SSIM drops from 0.80 to 0.25 with photographic tiles due to texture interference
   - **Geometric images**: SSIM maintains ~0.79 across both tile types, indicating robustness to geometric patterns
   - **High contrast images**: Minimal SSIM difference (0.82 vs 0.82), showing good adaptability

2. **Adaptive Grid Consistency**: Adaptive segmentation maintains its advantages across both tile types:
   - **Geometric patterns**: 28.6% (colored) and 23.4% (photographic) MSE improvement
   - **High contrast**: 79.7% (colored) and 79.2% (photographic) MSE improvement
   - **Smooth gradients**: Slight performance degradation in both cases

3. **Application-Specific Optimization**: Results suggest optimal tile selection depends on input content:
   - **Smooth gradients**: Colored squares provide better structural preservation
   - **Complex scenes**: Photographic tiles offer more natural appearance despite lower SSIM scores
   - **High contrast patterns**: Both tile types perform well with adaptive segmentation

### 3.2 Implementation Strengths

- **Computational Efficiency**: Optimized color matching using vectorized operations
- **User Experience**: Intuitive Gradio interface with real-time feedback and tabbed organization
- **Flexibility**: Support for both custom image tiles and colored squares with easy switching
- **Advanced Tile Management**: Professional-grade tile upload system with preview, validation, and batch processing
- **Robustness**: Comprehensive error handling, input validation, and automatic image preprocessing

### 3.3 Limitations and Future Work

Current limitations include:
- Color space matching limited to RGB (could explore LAB or HSV)
- No consideration of tile texture or pattern information
- Sequential processing limits scalability for large images

Future enhancements could include:
- Multi-threaded tile matching for performance
- Machine learning-based tile selection
- Support for non-square tiles and irregular grids

## 4. Conclusion

This implementation successfully demonstrates a comprehensive image mosaic generation system that balances quality, performance, and usability. The adaptive grid segmentation algorithm provides measurable improvements over fixed approaches, while the interactive interface enables effective parameter exploration. The quantitative evaluation framework ensures objective quality assessment, making this system suitable for both artistic applications and algorithm research.

The combination of technical sophistication and user-friendly design creates a robust platform for image mosaic generation that meets all specified requirements while providing opportunities for future enhancement.