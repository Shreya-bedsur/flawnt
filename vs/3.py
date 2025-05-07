import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Define season palette and recommendations
SEASON_PALETTES = {
    "Winter": ["#1c1f4c", "#5d6d7e", "#641e16"],
    "Summer": ["#f5b7b1", "#aed6f1", "#f9e79f"],
    "Autumn": ["#a04000", "#784212", "#b9770e"],
    "Spring": ["#f7dc6f", "#58d68d", "#f1948a"]
}
RECOMMENDATIONS = {
    "Winter": "Bold reds, cool-toned foundations, silver highlighters.",
    "Summer": "Rose pinks, light BB creams, subtle shimmer.",
    "Autumn": "Terracotta lipsticks, warm bronzers, earthy shadows.",
    "Spring": "Peachy tones, dewy finishes, coral blush."
}

class FlawntApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flawnt: Seasonal Color & Beauty Analyzer")
        self.image = None
        self.tk_image = None
        self.cropped_image = None

        # UI
        self.canvas = tk.Canvas(root, width=500, height=500, bg='white')
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Capture from Webcam", command=self.capture_webcam).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Crop", command=self.activate_crop).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="Analyze", command=self.analyze_image).grid(row=0, column=3, padx=5)

        self.result_label = tk.Label(root, text="", font=('Arial', 12), wraplength=400, justify='left')
        self.result_label.pack(pady=10)

        # Cropping
        self.start_x = self.start_y = None
        self.rect_id = None
        self.canvas.bind("<ButtonPress-1>", self.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_crop_end)
        self.cropping = False

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            self.display_image(self.image)

    def capture_webcam(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image(self.image)

    def display_image(self, img):
        img_resized = cv2.resize(img, (500, 500))
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(img_resized))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def activate_crop(self):
        if self.image is not None:
            self.cropping = True
            messagebox.showinfo("Crop", "Click and drag to crop the image.")

    def on_crop_start(self, event):
        if self.cropping:
            self.start_x, self.start_y = event.x, event.y

    def on_crop_drag(self, event):
        if self.cropping and self.start_x and self.start_y:
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='red')

    def on_crop_end(self, event):
        if self.cropping:
            end_x, end_y = event.x, event.y
            x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
            x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)

            scale_x = self.image.shape[1] / 500
            scale_y = self.image.shape[0] / 500
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            self.cropped_image = self.image[y1:y2, x1:x2]
            self.display_image(self.cropped_image)
            self.cropping = False

    def analyze_image(self):
        if self.cropped_image is None:
            messagebox.showerror("Error", "Please crop a region first.")
            return
        skin = self.detect_skin(self.cropped_image)
        dominant_color = self.get_dominant_color(skin)
        season = self.classify_season(dominant_color)
        color_hex = self.rgb_to_hex(dominant_color)
        recommendation = RECOMMENDATIONS.get(season, "Try neutral tones.")
        self.result_label.config(
            text=f"Detected Season: {season}\nDominant Skin Tone: {color_hex}\nRecommended Cosmetics:\n{recommendation}"
        )

    def detect_skin(self, img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        return cv2.bitwise_and(img, img, mask=mask)

    def get_dominant_color(self, img, k=3):
        img = img.reshape((-1, 3))
        img = img[np.any(img != [0, 0, 0], axis=1)]  # remove black
        if img.size == 0:
            return (128, 128, 128)
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(img)
        return tuple(map(int, kmeans.cluster_centers_[0]))

    def classify_season(self, rgb):
        r, g, b = rgb
        if r > 200 and g > 150:
            return "Spring"
        elif r > 150 and g < 100:
            return "Autumn"
        elif b > 150:
            return "Winter"
        else:
            return "Summer"

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % rgb

if __name__ == "__main__":
    root = tk.Tk()
    app = FlawntApp(root)
    root.mainloop()
