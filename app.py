import gradio as gr
import os
from enhanced_mosaic_generator import EnhancedMosaicGenerator
from image_preprocessing import create_sample_tiles, preprocess_tiles_folder
import json


def create_mosaic_interface(input_image, tile_size, grid_width, grid_height,
                           use_adaptive_grid, use_colored_tiles,
                           color_quantization):
    # Main function to create the mosaic
    if input_image is None:
        return None, "Please upload an image.", "{}"

    try:
        # Setup the generator
        generator = EnhancedMosaicGenerator(tile_size=int(tile_size))

        # Check if tiles exist, create samples if not
        tiles_folder = "tiles"
        if (not os.path.exists(tiles_folder) or
                len(os.listdir(tiles_folder)) == 0):
            create_sample_tiles(tiles_folder)
            status_msg = "No tiles found. Created sample tiles."
        else:
            status_msg = f"Using {len(os.listdir(tiles_folder))} tiles."

        # Load the tile images
        if not use_colored_tiles:
            generator.load_tiles(tiles_folder)
            if len(generator.tiles) == 0:
                return None, "No valid tiles found.", "{}"

        # Create mosaic
        output_path = "output_mosaic.jpg"
        grid_size = (None if grid_width <= 0 or grid_height <= 0
                     else (int(grid_width), int(grid_height)))

        result_path, metrics = generator.create_mosaic(
            input_image,
            output_path,
            grid_size=grid_size,
            use_adaptive_grid=use_adaptive_grid,
            use_colored_tiles=use_colored_tiles,
            color_quantization=int(color_quantization),
            base_grid_size=32
        )

        # Put the metrics in a nice format
        metrics_text = json.dumps({
            "MSE": round(metrics['mse'], 2),
            "SSIM": round(metrics['ssim'], 4),
            "PSNR": round(metrics['psnr'], 2)
        }, indent=2)

        return result_path, f"Mosaic created. {status_msg}", metrics_text

    except Exception as e:
        return None, f"Error: {str(e)}", "{}"


def setup_tiles_interface(uploaded_files):
    # Deal with uploaded tile files
    if not uploaded_files:
        return "No files uploaded."

    tiles_folder = "tiles"
    os.makedirs(tiles_folder, exist_ok=True)

    saved_count = 0
    for file_info in uploaded_files:
        try:
            filename = os.path.basename(file_info.name)
            destination = os.path.join(tiles_folder, filename)

            with open(file_info.name, 'rb') as src, \
                 open(destination, 'wb') as dst:
                dst.write(src.read())

            saved_count += 1
        except Exception as e:
            print(f"Error saving {file_info.name}: {e}")

    # Process all the uploaded tiles
    try:
        preprocess_tiles_folder(tiles_folder)
        return f"Uploaded and processed {saved_count} tiles."
    except Exception as e:
        return f"Had trouble processing tiles: {e}"


def get_tiles_gallery():
    # Get all the tiles for the gallery
    tiles_folder = "tiles"
    if not os.path.exists(tiles_folder):
        return []

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    tile_files = [f for f in os.listdir(tiles_folder)
                  if f.lower().endswith(supported_formats)]

    tile_paths = [os.path.join(tiles_folder, f) for f in sorted(tile_files)]
    return tile_paths



def refresh_tiles():
    # Refresh the tiles gallery
    return get_tiles_gallery()


def enhanced_setup_tiles_interface(uploaded_files):
    # Upload tiles and refresh gallery right away
    result = setup_tiles_interface(uploaded_files)
    return result, get_tiles_gallery()


# Create basic Gradio interface
with gr.Blocks(title="Image Mosaic Generator") as demo:
    gr.Markdown("# Image Mosaic Generator")
    gr.Markdown("Assignment implementation for image mosaic creation.")

    with gr.Tab("Create Mosaic"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Upload Image")

                # Example images - dynamically scan examples folder
                examples_folder = "examples"
                if os.path.exists(examples_folder):
                    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
                    example_files = [f for f in os.listdir(examples_folder)
                                   if f.lower().endswith(supported_formats)]
                    available_examples = [os.path.join(examples_folder, f)
                                        for f in sorted(example_files)]
                else:
                    available_examples = []

                if available_examples:
                    gr.Markdown("### Example Images")
                    example_selector = gr.Radio(
                        choices=[(os.path.basename(img), img)
                               for img in available_examples],
                        label="Select Example",
                        value=None
                    )

                    def load_example(selected):
                        return selected if selected else None

                    example_selector.change(
                        fn=load_example,
                        inputs=[example_selector],
                        outputs=[input_image]
                    )

                gr.Markdown("### Settings")
                tile_size = gr.Slider(minimum=8, maximum=64, value=32, step=8,
                                    label="Tile Size")

                with gr.Row():
                    grid_width = gr.Number(value=0, label="Grid Width (0=auto)")
                    grid_height = gr.Number(value=0, label="Grid Height (0=auto)")

                use_adaptive_grid = gr.Checkbox(
                    label="Use Adaptive Grid (8x8, 16x16, 32x32)",
                    info="Automatically subdivide based on image complexity")

                use_colored_tiles = gr.Checkbox(
                    label="Use Colored Tiles",
                    info="Use solid colors instead of image tiles")

                color_quantization = gr.Slider(
                    minimum=0, maximum=32, value=0, step=4,
                    label="Color Quantization (0=off)",
                    info="Reduce number of colors")

                create_btn = gr.Button("Create Mosaic", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Mosaic Result")
                status_text = gr.Textbox(label="Status", interactive=False)
                metrics_text = gr.Code(label="Performance Metrics",
                                     language="json")

        create_btn.click(
            fn=create_mosaic_interface,
            inputs=[input_image, tile_size, grid_width, grid_height,
                    use_adaptive_grid, use_colored_tiles, color_quantization],
            outputs=[output_image, status_text, metrics_text]
        )

    with gr.Tab("Manage Tiles"):
        gr.Markdown("### Tile Management")
        gr.Markdown("Upload, preview, and manage your mosaic tiles.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Upload New Tiles")
                tile_upload = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Upload Tile Images"
                )
                upload_btn = gr.Button("Process Tiles")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

                gr.Markdown("#### Tile Actions")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Gallery")
                    reset_btn = gr.Button("Reset to Default", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("#### Current Tiles")
                tiles_gallery = gr.Gallery(
                    label="Tiles Preview",
                    show_label=True,
                    elem_id="tiles_gallery",
                    columns=4,
                    rows=3,
                    height="auto"
                )


        # Connect all the buttons to functions
        upload_btn.click(
            fn=enhanced_setup_tiles_interface,
            inputs=[tile_upload],
            outputs=[upload_status, tiles_gallery]
        )

        refresh_btn.click(
            fn=refresh_tiles,
            outputs=[tiles_gallery]
        )

        def simple_reset_tiles():
            # Reset tiles to default colored squares
            tiles_folder = "tiles"
            try:
                # Remove all existing tiles
                if os.path.exists(tiles_folder):
                    for f in os.listdir(tiles_folder):
                        os.remove(os.path.join(tiles_folder, f))
                # Create default tiles
                create_sample_tiles(tiles_folder)
                return get_tiles_gallery()
            except Exception:
                return get_tiles_gallery()

        reset_btn.click(
            fn=simple_reset_tiles,
            outputs=[tiles_gallery]
        )

        # Show tiles when app first loads
        demo.load(
            fn=lambda: refresh_tiles(),
            outputs=[tiles_gallery]
        )

    with gr.Tab("Instructions"):
        gr.Markdown("""
        ## How to Use

        1. **Upload Image**: Choose an image file or select an example
        2. **Adjust Settings**:
           - Tile Size: Size of mosaic pieces (8-64 pixels)
           - Grid Size: Leave at 0 for automatic sizing
           - Adaptive Grid: Enable for variable grid sizes (8x8, 16x16, 32x32)
           - Colored Tiles: Use solid colors instead of images
           - Color Quantization: Reduce colors for artistic effects
        3. **Create Mosaic**: Click button to generate result
        4. **View Metrics**: Check MSE, SSIM, and PSNR performance scores

        ## Performance Metrics
        - MSE: Mean Squared Error (lower is better)
        - SSIM: Structural Similarity (0-1, higher is better)
        - PSNR: Peak Signal-to-Noise Ratio (higher is better)

        ## Tile Management
        - **View Current Tiles**: See all available tiles in the gallery
        - **Upload Custom Tiles**: Add your own images as mosaic tiles
        - **Reset Tiles**: Return to default colored squares
        - **Refresh Gallery**: Update the tile preview
        """)


if __name__ == "__main__":
    # Make sure we have the right folders
    os.makedirs("tiles", exist_ok=True)
    os.makedirs("examples", exist_ok=True)

    # If no tiles, make some default ones
    if len(os.listdir("tiles")) == 0:
        create_sample_tiles("tiles")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )