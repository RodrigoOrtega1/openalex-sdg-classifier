import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import numpy as np

load_dotenv('./.env')
IMAGES_PATH = os.getenv('IMAGES_PATH')

goal_names = {
    1: "No poverty",
    2: "Zero hunger",
    3: "Good health and well-being",
    4: "Quality Education",
    5: "Gender equality",
    6: "Clean water and sanitation",
    7: "Affordable and clean energy",
    8: "Decent work and economic growth",
    9: "Industry, innovation and infrastructure",
    10: "Reduced inequalities",
    11: "Sustainable cities and communities",
    12: "Responsible consumption and production",
    13: "Climate action",
    14: "Life below water",
    15: "Life on Land",
    16: "Peace, Justice and strong institutions",
    17: "Partnerships for the goals"
}


def offset_sdg_image(x, y, sdg_number, bar_is_too_short, ax):
    """
    Adds an SDG image to a vertical bar plot.
    Args:
        x (int): The x-coordinate of the bar center.
        y (float): The height of the bar.
        sdg_number (int): The SDG number to determine which image to use.
        bar_is_too_short (bool): Whether the bar is too short to contain the image comfortably.
        ax (matplotlib.axes.Axes): The axes to which the image will be added.
    Returns:
        None
    """
    image_path = os.path.join(IMAGES_PATH, f"E_SDG_icons-{sdg_number:02d}.jpg")

    try:
        with Image.open(image_path) as pil_img:
            pil_img = pil_img.convert('RGBA')
            img_arr = np.array(pil_img)
    except Exception as e:
        print(f"Warning: Failed to load image '{image_path}': {e}")
        return

    im = OffsetImage(img_arr, zoom=0.1)
    im.image.axes = ax

    if bar_is_too_short:
        y = 0

    y_offset = -25
    ab = AnnotationBbox(im, (x, y), xybox=(0, y_offset), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)


def _get_dominant_color(image_path):
    """
    Finds the dominant color of an image.
    Args:
        image_path (str): The path to the image file.
    Returns:
        tuple: The dominant color as an (R, G, B) tuple scaled to 0-1, or None if the image cannot be processed.
    """
    try:
        with Image.open(image_path) as img:
            img = img.resize((1, 1), Image.Resampling.LANCZOS)
            color = img.getpixel((0, 0))
            return (color[0] / 255, color[1] / 255, color[2] / 255)
    except Exception as e:
        print(f"Warning: Failed to process image for color '{image_path}': {e}")
        return (0.5, 0.5, 0.5) # Default to gray if image fails


def plot_sdg_barplot_with_images(predictions):
    """
    Generates a bar plot of SDG predictions with images and returns it as a base64 string.
    Args:
        predictions (list of dict): List of prediction dictionaries with 'sdg' and 'prediction' keys.
    Returns:
        str: Base64-encoded PNG image of the bar plot.
    """
    # Data Preparation from predictions
    sdg_numbers = list(goal_names.keys())
    labels = [f"SDG {i}" for i in sdg_numbers]
    
    prediction_map = {int(p['sdg']['code']): p['prediction'] for p in predictions}
    values = [prediction_map.get(i, 0) for i in sdg_numbers]

    # Get bar colors from images
    bar_colors = []
    for sdg_num in sdg_numbers:
        image_path = os.path.join(IMAGES_PATH, f"E_SDG_icons-{sdg_num:02d}.jpg")
        bar_colors.append(_get_dominant_color(image_path))

    fig = Figure(figsize=(15, 8))
    ax = fig.subplots()
    width = 0.8
    ax.bar(x=labels, height=values, width=width, align='center', alpha=0.8, color=bar_colors)

    # 3. Remove original x-tick labels
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)

    # 4. Add images as x-tick labels
    for i, sdg_num in enumerate(sdg_numbers):
        image_path = os.path.join(IMAGES_PATH, f"E_SDG_icons-{sdg_num:02d}.jpg")
        try:
            with Image.open(image_path) as pil_img:
                img_arr = np.array(pil_img.convert('RGBA'))
                im = OffsetImage(img_arr, zoom=0.1)
                im.image.axes = ax
                ab = AnnotationBbox(im, (i, 0), xybox=(0, -35), frameon=False,
                                    xycoords='data', boxcoords="offset points", pad=0)
                ax.add_artist(ab)
        except Exception as e:
            print(f"Warning: Failed to load image for label '{image_path}': {e}")

    ax.set_ylabel('Prediction Score')
    ax.set_title('SDG Classification Predictions')
    fig.subplots_adjust(bottom=0.2)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return data