#!/usr/bin/env python3
"""Performance testing script for mosaic generator."""

import os
import sys
from enhanced_mosaic_generator import EnhancedMosaicGenerator
from image_preprocessing import create_sample_tiles
from create_examples import create_test_images
import json

def run_performance_tests():
    """Run comprehensive performance tests."""
    print("=== Image Mosaic Generator Performance Test ===\n")

    # Ensure test images exist
    if not os.path.exists("examples"):
        print("Creating test images...")
        create_test_images()

    # Ensure tiles exist
    if not os.path.exists("tiles") or len(os.listdir("tiles")) == 0:
        print("Creating sample tiles...")
        create_sample_tiles("tiles")

    # Initialize generator
    generator = EnhancedMosaicGenerator(tile_size=32)
    generator.load_tiles("tiles")

    # Test images
    test_images = [
        ("examples/gradient.jpg", "Gradient Image (Smooth Transitions)"),
        ("examples/geometric.jpg", "Geometric Image (Sharp Boundaries)"),
        ("examples/high_contrast.jpg", "High Contrast Image (Checkerboard)")
    ]

    results = {}

    for image_path, description in test_images:
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping...")
            continue

        print(f"\n--- Testing: {description} ---")
        results[description] = {}

        # Test 1: Fixed 32x32 grid
        print("Testing Fixed 32x32 Grid...")
        try:
            output_path = f"test_output_fixed_{os.path.basename(image_path)}"
            result_path, metrics = generator.create_mosaic(
                image_path,
                output_path,
                grid_size=(32, 32),
                use_adaptive_grid=False,
                use_colored_tiles=False
            )

            results[description]["Fixed 32x32"] = {
                "MSE": round(metrics['mse'], 1),
                "SSIM": round(metrics['ssim'], 4),
                "PSNR": round(metrics['psnr'], 2)
            }

            print(f"  MSE: {metrics['mse']:.1f}")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f}")

        except Exception as e:
            print(f"  Error in fixed grid test: {e}")
            results[description]["Fixed 32x32"] = {"Error": str(e)}

        # Test 2: Adaptive grid
        print("Testing Adaptive Grid...")
        try:
            output_path = f"test_output_adaptive_{os.path.basename(image_path)}"
            result_path, metrics = generator.create_mosaic(
                image_path,
                output_path,
                grid_size=None,
                use_adaptive_grid=True,
                use_colored_tiles=False,
                base_grid_size=32
            )

            results[description]["Adaptive Grid"] = {
                "MSE": round(metrics['mse'], 1),
                "SSIM": round(metrics['ssim'], 4),
                "PSNR": round(metrics['psnr'], 2)
            }

            print(f"  MSE: {metrics['mse']:.1f}")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f}")

        except Exception as e:
            print(f"  Error in adaptive grid test: {e}")
            results[description]["Adaptive Grid"] = {"Error": str(e)}

    # Save results to JSON
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Test Complete ===")
    print("Results saved to test_results.json")
    print("\nSummary:")

    for image_name, tests in results.items():
        print(f"\n{image_name}:")
        for test_name, metrics in tests.items():
            if "Error" not in metrics:
                print(f"  {test_name}: MSE={metrics['MSE']}, SSIM={metrics['SSIM']}, PSNR={metrics['PSNR']}")
            else:
                print(f"  {test_name}: Failed - {metrics['Error']}")

    return results

if __name__ == "__main__":
    run_performance_tests()