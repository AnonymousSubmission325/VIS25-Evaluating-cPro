import os
from PIL import Image

# Define the folder paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "plots")
output_image = os.path.join(script_dir, "combined_output.jpg")

ordered_filenames = [
    "wine_-_Simulated_Annealing_cPro_projection.png",
    "wine_-_Adam_cPro_projection.png",
    "wine_-_PSO_cPro_projection.png",
    "wine_-_L-BFGS_cPro_projection.png",
    "wine_-_SOM_projection.png",
    "wine_-_Spring-Force_Radial_Projection_projection.png",
    "wine_-_Isomap_2D_projection.png",
    "wine_-_MDS_2D_projection.png",
    "wine_-_PCA_2D_projection.png",
    "wine_-_t-SNE_2D_projection.png",
    "wine_-_UMAP_2D_projection.png",
    # "wine_-_MDS_1D_projection.png",
    # "wine_-_PCA_1D_projection.png"
]

# Customize the crop percentage for each edge
crop_percentages = {
    "left": 0.12,   # 10% from the left
    "right": 0.04,  # 10% from the right
    "top": 0.06,   # 5% from the top
    "bottom": 0.08  # 10% from the bottom
}

# List to store the processed images
processed_images = []

# Load and process each image
for filename in ordered_filenames:
    file_path = os.path.join(input_folder, filename)
    if os.path.exists(file_path):
        # Open image
        img = Image.open(file_path)
        width, height = img.size

        # Calculate crop dimensions based on specified percentages
        left = width * crop_percentages["left"]
        right = width * (1 - crop_percentages["right"])
        top = height * crop_percentages["top"]
        bottom = height * (1 - crop_percentages["bottom"])
        
        # Crop the image based on the specified edges
        cropped_img = img.crop((left, top, right, bottom))

        # Append the processed image to the list
        processed_images.append(cropped_img)
    else:
        print(f"[WARNING] {filename} not found in {input_folder}.")

# Concatenate all images horizontally
if processed_images:
    total_width = sum(img.width for img in processed_images)
    max_height = max(img.height for img in processed_images)

    combined_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in processed_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    combined_image.save(output_image)
    print(f"[INFO] Combined image saved as {output_image}")
else:
    print("[ERROR] No images were processed.")
