import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')  
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

image = cv2.imread('s3.jpg')

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

max_height = 400
if image.shape[0] > max_height:
    scaling_factor = max_height / image.shape[0]
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

original_image = image.copy()

brightness = 0
contrast = 0
blur = 0
median = 0
sharpen = False
hist_eq = False
edge = False

def get_adjusted_image():
    global brightness, contrast, blur, median, sharpen, hist_eq, edge
    alpha = 1 + (contrast / 100.0)  
    beta = brightness               

    adjusted = cv2.convertScaleAbs(original_image, alpha=alpha, beta=beta)

    if blur > 0:
        ksize = (2 * blur + 1, 2 * blur + 1)  
        adjusted = cv2.GaussianBlur(adjusted, ksize, 0)

    if median > 0:
        ksize = 2 * median + 1  
        adjusted = cv2.medianBlur(adjusted, ksize)

    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        adjusted = cv2.filter2D(adjusted, -1, kernel)

    if hist_eq:
        ycrcb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        adjusted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if edge:
        adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        adjusted_edges = cv2.Canny(adjusted_gray, 100, 200)
        adjusted = cv2.cvtColor(adjusted_edges, cv2.COLOR_GRAY2BGR)

    return adjusted

def update_image(*args):
    adjusted = get_adjusted_image()
    cv2.imshow('Image Enhancement', adjusted)

    update_histogram(adjusted)

def update_histogram(image):
    ax.clear()  

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    ax.plot(hist, color='black')
    ax.set_xlim([0, 256])
    ax.set_title('Grayscale Histogram')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')

    canvas.draw()

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

def save_image():
    adjusted = get_adjusted_image()
    cv2.imwrite('enhanced_image.jpg', adjusted)
    print("Image saved as 'enhanced_image.jpg'.")

def reset_settings():
    brightness_scale.set(100)
    contrast_scale.set(100)
    blur_scale.set(0)
    median_scale.set(0)
    sharpen_var.set(False)
    hist_eq_var.set(False)
    edge_var.set(False)
    update_image()

def close_program():
    root.destroy()
    cv2.destroyAllWindows()
    plt.close('all')

root = tk.Tk()
root.title("Image Enhancement Tool")

controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

hist_frame = tk.Frame(root)
hist_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

brightness_scale = tk.Scale(controls_frame, from_=0, to=200, orient=tk.HORIZONTAL, label='Brightness', command=on_brightness_change)
brightness_scale.set(100)
brightness_scale.pack(fill=tk.X, padx=5, pady=5)

contrast_scale = tk.Scale(controls_frame, from_=0, to=200, orient=tk.HORIZONTAL, label='Contrast', command=on_contrast_change)
contrast_scale.set(100)
contrast_scale.pack(fill=tk.X, padx=5, pady=5)

blur_scale = tk.Scale(controls_frame, from_=0, to=10, orient=tk.HORIZONTAL, label='Gaussian Blur', command=on_blur_change)
blur_scale.pack(fill=tk.X, padx=5, pady=5)

median_scale = tk.Scale(controls_frame, from_=0, to=10, orient=tk.HORIZONTAL, label='Median Blur', command=on_median_change)
median_scale.pack(fill=tk.X, padx=5, pady=5)

sharpen_var = tk.BooleanVar()
sharpen_check = tk.Checkbutton(controls_frame, text='Sharpen', variable=sharpen_var, command=on_sharpen_toggle)
sharpen_check.pack(anchor='w', padx=5, pady=2)

hist_eq_var = tk.BooleanVar()
hist_eq_check = tk.Checkbutton(controls_frame, text='Histogram Equalization', variable=hist_eq_var, command=on_hist_eq_toggle)
hist_eq_check.pack(anchor='w', padx=5, pady=2)

edge_var = tk.BooleanVar()
edge_check = tk.Checkbutton(controls_frame, text='Edge Detection', variable=edge_var, command=on_edge_toggle)
edge_check.pack(anchor='w', padx=5, pady=2)

button_frame = tk.Frame(controls_frame)
button_frame.pack(fill=tk.X, padx=5, pady=5)

save_button = tk.Button(button_frame, text='Save Image', command=save_image)
save_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

reset_button = tk.Button(button_frame, text='Reset', command=reset_settings)
reset_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

exit_button = tk.Button(button_frame, text='Exit', command=close_program)
exit_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

fig = Figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)

canvas = FigureCanvasTkAgg(fig, master=hist_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

cv2.namedWindow('Image Enhancement', cv2.WINDOW_AUTOSIZE)
update_image()

root.protocol("WM_DELETE_WINDOW", close_program)

root.mainloop()
