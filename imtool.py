import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('s1.jpg')

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Optional: Resize if larger than a given max height
max_height = 400
if image.shape[0] > max_height:
    scaling_factor = max_height / image.shape[0]
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

original_image = image.copy()

# Global variables
brightness = 0
contrast = 0
blur = 0
median = 0
sharpen = False
hist_eq = False
edge = False

# New global variables for additional features
bilateral_d = 0          # Controls the diameter for bilateral filter
b_balance = 0            # Blue channel adjustment
g_balance = 0            # Green channel adjustment
r_balance = 0            # Red channel adjustment
side_by_side_var = False # Whether to show side-by-side comparison

def get_adjusted_image():
    """
    Apply all selected enhancements/filters to the original image.
    Returns the processed image.
    """
    global brightness, contrast, blur, median
    global sharpen, hist_eq, edge
    global bilateral_d, b_balance, g_balance, r_balance

    # --- Brightness & Contrast ---
    alpha = 1 + (contrast / 100.0)  # Contrast scaling factor
    beta = brightness               # Brightness offset
    adjusted = cv2.convertScaleAbs(original_image, alpha=alpha, beta=beta)

    # --- Gaussian Blur ---
    if blur > 0:
        ksize = (2 * blur + 1, 2 * blur + 1)
        adjusted = cv2.GaussianBlur(adjusted, ksize, 0)

    # --- Median Blur ---
    if median > 0:
        ksize = 2 * median + 1
        adjusted = cv2.medianBlur(adjusted, ksize)

    # --- Bilateral Filter ---
    if bilateral_d > 0:
        # sigmaColor, sigmaSpace can be tweaked as needed
        adjusted = cv2.bilateralFilter(adjusted, d=bilateral_d, sigmaColor=75, sigmaSpace=75)

    # --- Sharpen ---
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        adjusted = cv2.filter2D(adjusted, -1, kernel)

    # --- Histogram Equalization (on Y channel) ---
    if hist_eq:
        ycrcb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        adjusted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # --- Edge Detection ---
    if edge:
        adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        adjusted_edges = cv2.Canny(adjusted_gray, 100, 200)
        adjusted = cv2.cvtColor(adjusted_edges, cv2.COLOR_GRAY2BGR)

    # --- Color Balance Adjustments ---
    b_channel, g_channel, r_channel = cv2.split(adjusted.astype(np.int16))
    b_channel = np.clip(b_channel + b_balance, 0, 255)
    g_channel = np.clip(g_channel + g_balance, 0, 255)
    r_channel = np.clip(r_channel + r_balance, 0, 255)
    adjusted = cv2.merge([b_channel.astype(np.uint8),
                          g_channel.astype(np.uint8),
                          r_channel.astype(np.uint8)])

    return adjusted

def update_image(*args):
    """
    Update the displayed image and histogram when any parameter changes.
    If side-by-side is selected, shows both original and adjusted images together.
    """
    global side_by_side_var
    adjusted = get_adjusted_image()

    if side_by_side_var:
        # Ensure both images are the same size for side-by-side
        h, w = original_image.shape[:2]
        adjusted_resized = cv2.resize(adjusted, (w, h))

        side_by_side = np.hstack((original_image, adjusted_resized))
        cv2.imshow('Image Enhancement', side_by_side)
        update_histogram(adjusted_resized)
    else:
        cv2.imshow('Image Enhancement', adjusted)
        update_histogram(adjusted)

def update_histogram(image):
    """
    Update the histogram plot for the given image (grayscale),
    with an improved look and feel.
    """
    ax.clear()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist.ravel()  # Convert to 1D array

    # Fill the area under the histogram
    ax.fill_between(range(256), hist, color='gray', alpha=0.75)

    ax.set_xlim([0, 256])
    ax.set_ylim([0, hist.max() * 1.1])  # Slightly above max for breathing room
    ax.set_title('Grayscale Histogram')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')

    # Optional: Add a grid
    ax.grid(True, linestyle='--', alpha=0.5)

    canvas.draw()

# ------------------------- Callbacks -------------------------

def on_brightness_change(val):
    global brightness
    brightness = int(val) - 100
    update_image()

def on_contrast_change(val):
    global contrast
    contrast = int(val) - 100
    update_image()

def on_blur_change(val):
    global blur
    blur = int(val)
    update_image()

def on_median_change(val):
    global median
    median = int(val)
    update_image()

def on_bilateral_change(val):
    global bilateral_d
    bilateral_d = int(val)
    update_image()

def on_sharpen_toggle():
    global sharpen
    sharpen = sharpen_var.get()
    update_image()

def on_hist_eq_toggle():
    global hist_eq
    hist_eq = hist_eq_var.get()
    update_image()

def on_edge_toggle():
    global edge
    edge = edge_var.get()
    update_image()

def on_side_by_side_toggle():
    global side_by_side_var
    side_by_side_var = side_by_side_check_var.get()
    update_image()

def on_b_balance_change(val):
    global b_balance
    b_balance = int(val) - 100
    update_image()

def on_g_balance_change(val):
    global g_balance
    g_balance = int(val) - 100
    update_image()

def on_r_balance_change(val):
    global r_balance
    r_balance = int(val) - 100
    update_image()

def save_image():
    adjusted = get_adjusted_image()
    cv2.imwrite('enhanced_image.jpg', adjusted)
    print("Image saved as 'enhanced_image.jpg'.")

def reset_settings():
    brightness_scale.set(100)
    contrast_scale.set(100)
    blur_scale.set(0)
    median_scale.set(0)
    bilateral_scale.set(0)
    sharpen_var.set(False)
    hist_eq_var.set(False)
    edge_var.set(False)
    side_by_side_check_var.set(False)

    b_balance_scale.set(100)
    g_balance_scale.set(100)
    r_balance_scale.set(100)

    update_image()

def close_program():
    root.destroy()
    cv2.destroyAllWindows()
    plt.close('all')

# ------------------------- UI Setup -------------------------
root = tk.Tk()
root.title("Image Enhancement Tool")

# Frames for layout
controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

hist_frame = tk.Frame(root)
hist_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# -------------------- Brightness & Contrast --------------------
brightness_scale = tk.Scale(
    controls_frame, from_=0, to=200, orient=tk.HORIZONTAL,
    label='Brightness', command=on_brightness_change
)
brightness_scale.set(100)
brightness_scale.pack(fill=tk.X, padx=5, pady=5)

contrast_scale = tk.Scale(
    controls_frame, from_=0, to=200, orient=tk.HORIZONTAL,
    label='Contrast', command=on_contrast_change
)
contrast_scale.set(100)
contrast_scale.pack(fill=tk.X, padx=5, pady=5)

# -------------------- Gaussian & Median Blur --------------------
blur_scale = tk.Scale(
    controls_frame, from_=0, to=10, orient=tk.HORIZONTAL,
    label='Gaussian Blur', command=on_blur_change
)
blur_scale.pack(fill=tk.X, padx=5, pady=5)

median_scale = tk.Scale(
    controls_frame, from_=0, to=10, orient=tk.HORIZONTAL,
    label='Median Blur', command=on_median_change
)
median_scale.pack(fill=tk.X, padx=5, pady=5)

# -------------------- Bilateral Filter --------------------
bilateral_scale = tk.Scale(
    controls_frame, from_=0, to=10, orient=tk.HORIZONTAL,
    label='Bilateral Filter (d)', command=on_bilateral_change
)
bilateral_scale.pack(fill=tk.X, padx=5, pady=5)

# -------------------- Color Balance Sliders --------------------
b_balance_scale = tk.Scale(
    controls_frame, from_=0, to=200, orient=tk.HORIZONTAL,
    label='Blue Balance', command=on_b_balance_change
)
b_balance_scale.set(100)
b_balance_scale.pack(fill=tk.X, padx=5, pady=5)

g_balance_scale = tk.Scale(
    controls_frame, from_=0, to=200, orient=tk.HORIZONTAL,
    label='Green Balance', command=on_g_balance_change
)
g_balance_scale.set(100)
g_balance_scale.pack(fill=tk.X, padx=5, pady=5)

r_balance_scale = tk.Scale(
    controls_frame, from_=0, to=200, orient=tk.HORIZONTAL,
    label='Red Balance', command=on_r_balance_change
)
r_balance_scale.set(100)
r_balance_scale.pack(fill=tk.X, padx=5, pady=5)

# -------------------- Checkbuttons for toggles --------------------
sharpen_var = tk.BooleanVar()
sharpen_check = tk.Checkbutton(
    controls_frame, text='Sharpen', variable=sharpen_var,
    command=on_sharpen_toggle
)
sharpen_check.pack(anchor='w', padx=5, pady=2)

hist_eq_var = tk.BooleanVar()
hist_eq_check = tk.Checkbutton(
    controls_frame, text='Histogram Equalization', variable=hist_eq_var,
    command=on_hist_eq_toggle
)
hist_eq_check.pack(anchor='w', padx=5, pady=2)

edge_var = tk.BooleanVar()
edge_check = tk.Checkbutton(
    controls_frame, text='Edge Detection', variable=edge_var,
    command=on_edge_toggle
)
edge_check.pack(anchor='w', padx=5, pady=2)

# Side-by-side comparison toggle
side_by_side_check_var = tk.BooleanVar()
side_by_side_check = tk.Checkbutton(
    controls_frame, text='Side-by-Side Comparison',
    variable=side_by_side_check_var,
    command=on_side_by_side_toggle
)
side_by_side_check.pack(anchor='w', padx=5, pady=2)

# -------------------- Buttons --------------------
button_frame = tk.Frame(controls_frame)
button_frame.pack(fill=tk.X, padx=5, pady=5)

save_button = tk.Button(button_frame, text='Save Image', command=save_image)
save_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

reset_button = tk.Button(button_frame, text='Reset', command=reset_settings)
reset_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

exit_button = tk.Button(button_frame, text='Exit', command=close_program)
exit_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

# -------------------- Histogram Figure --------------------
fig = Figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=hist_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# -------------------- Initial Display --------------------
cv2.namedWindow('Image Enhancement', cv2.WINDOW_AUTOSIZE)
update_image()

root.protocol("WM_DELETE_WINDOW", close_program)
root.mainloop()
