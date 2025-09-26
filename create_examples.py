from PIL import Image, ImageDraw
import os


def create_test_images():
    # Make some test images to try the mosaic on
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)

    def create_gradient_image():
        # Make a smooth color gradient
        size = (400, 300)
        image = Image.new('RGB', size)
        draw = ImageDraw.Draw(image)

        for x in range(size[0]):
            for y in range(size[1]):
                r = int(255 * x / size[0])  # Horizontal red gradient
                g = int(255 * y / size[1])  # Vertical green gradient
                b = int(255 * (x + y) / (size[0] + size[1]))  # Diagonal blue
                draw.point((x, y), (r, g, b))

        return image

    def create_geometric_image():
        # Make some geometric shapes
        size = (400, 300)
        image = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(image)

        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for i in range(6):
            x = (i % 3) * 130 + 20
            y = (i // 3) * 120 + 30
            draw.rectangle([x, y, x+100, y+80], fill=colors[i])

        return image

    def create_high_contrast_image():
        # Make a checkerboard pattern
        size = (400, 300)
        image = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(image)

        # Black and white squares
        square_size = 40
        for x in range(0, size[0], square_size):
            for y in range(0, size[1], square_size):
                if (x // square_size + y // square_size) % 2 == 0:
                    draw.rectangle([x, y, x + square_size, y + square_size],
                                 fill='black')

        return image

    # Just make the 3 test images we need
    images = {
        'gradient.jpg': create_gradient_image(),
        'geometric.jpg': create_geometric_image(),
        'high_contrast.jpg': create_high_contrast_image()
    }

    for filename, img in images.items():
        img.save(os.path.join(examples_dir, filename))
        print(f"Created {filename}")

    print(f"Created {len(images)} test images in {examples_dir}/")


if __name__ == "__main__":
    create_test_images()