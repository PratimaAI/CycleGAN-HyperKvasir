from PIL import Image, ImageDraw
import os
import random

# Updated folder paths
folder_path = 'testA'
new_folder_path = 'testB'

# Create the destination folder if it doesn't exist
os.makedirs(new_folder_path, exist_ok=True)

# Process the images
files = os.listdir(folder_path)
for file in files:
    # Check if filename ends with a recognized image extension
    if file.endswith(('.jpg', '.png', '.bmp')):  # Add more extensions as needed
        img = Image.open(os.path.join(folder_path, file))
        draw = ImageDraw.Draw(img)

        # Generate random radius and coordinates for the circle
        width, height = img.size
        radius = random.randint(10, 100)  # Adjust minimum and maximum radius as needed
        x = random.randint(radius, width - radius)
        y = random.randint(radius, height - radius)

        # Draw the circle with random size
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue')

        # Save the modified image
        img.save(os.path.join(new_folder_path, 'modified_' + file))

    else:
        print(f"Skipping non-image file: {file}")  # Optionally log skipped files


