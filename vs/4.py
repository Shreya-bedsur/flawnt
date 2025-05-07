import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans
import ttkbootstrap as tb

# Seasonal types and color mappings
season_types = {
    'Winter': ['Cool', 'Deep', 'Bright'],
    'Summer': ['Cool', 'Light', 'Soft'],
    'Autumn': ['Warm', 'Deep', 'Soft'],
    'Spring': ['Warm', 'Light', 'Bright']
}

season_colors = {
    'Winter': '#7F7FFF',
    'Summer': '#FFB6C1',
    'Autumn': '#D2691E',
    'Spring': '#FFD700'
}

# Tooltips for results
season_tips = {
    'Winter': 'You look great in bold, cool colors like black, white, and jewel tones.',
    'Summer': 'Soft, cool shades like lavender, rose, and pastels complement you well.',
    'Autumn': 'Earth tones such as olive, rust, and warm browns enhance your glow.',
    'Spring': 'Clear, warm colors like coral, peach, and light turquoise suit you best.'
}


class FlawntApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flawnt - Skin Tone & Seasonal Color Analyzer")
        self.style = tb.Style("superhero")  # Use ttkbootstrap theme

        self.root.geometry("900x600")
        self.root.resizable(False, False)

        self.image_label = None
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.cropped_image = None
        self.crop_coords = None

        # UI Elements
        self.setup_ui()

    def setup_ui(self):
        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.pack(fill="both", expand=True)

        title = ttk.Label(self.frame, text="Flawnt", font=("Helvetica", 26, "bold"), foreground="turquoise")
        title.pack(pady=10)

        self.image_label = ttk.Label(self.frame)
        self.image_label.pack(pady=10)

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Upload Image", command=self.upload_image, bootstyle="info").grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="Capture from Camera", command=self.capture_image, bootstyle="info").grid(row=0, column=1, padx=10)
        ttk.Button(button_frame, text="Crop Image", command=self.crop_image, bootstyle="warning").grid(row=0, column=2, padx=10)
        ttk.Button(button_frame, text="Analyze Skin Tone", command=self.analyze_skin_tone, bootstyle="success").grid(row=0, column=3, padx=10)

        self.result_label = ttk.Label(self.frame, text="", font=("Helvetica", 14), wraplength=600)
        self.result_label.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(file_path).convert('RGB')
            self.display_image = self.original_image.copy()
            self.show_image(self.display_image)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        messagebox.showinfo("Capture", "Press 's' to capture image and 'q' to quit.")
        while True:
            ret, frame = cap.read()
            cv2.imshow('Capture Image - Press s to Save', frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite('captured_image.jpg', frame)
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        cv2.destroyAllWindows()
        self.image_path = 'captured_image.jpg'
        self.original_image = Image.open('captured_image.jpg').convert('RGB')
        self.display_image = self.original_image.copy()
        self.show_image(self.display_image)

    def show_image(self, img):
        img_resized = img.copy()
        img_resized.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def crop_image(self):
        if not self.display_image:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        crop_window = tk.Toplevel(self.root)
        crop_window.title("Crop Image")

        canvas = tk.Canvas(crop_window, width=500, height=500)
        canvas.pack()
        img = self.display_image.copy()
        img.thumbnail((500, 500))
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.image = img_tk

        self.start_x = self.start_y = self.end_x = self.end_y = 0

        def on_mouse_down(event):
            self.start_x, self.start_y = event.x, event.y

        def on_mouse_up(event):
            self.end_x, self.end_y = event.x, event.y
            cropped = img.crop((self.start_x, self.start_y, self.end_x, self.end_y))
            self.cropped_image = cropped
            self.show_image(cropped)
            crop_window.destroy()

        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)

    def analyze_skin_tone(self):
        img = self.cropped_image if self.cropped_image else self.display_image
        if img is None:
            messagebox.showerror("Error", "No image to analyze.")
            return

        img_np = np.array(img)
        img_np = img_np.reshape((-1, 3))
        kmeans = KMeans(n_clusters=3, random_state=42).fit(img_np)
        dominant_color = np.mean(kmeans.cluster_centers_, axis=0)

        r, g, b = dominant_color
        result = self.classify_skin_tone(r, g, b)

        if result:
            season, description = result
            color_hex = season_colors.get(season, "#CCCCCC")
            self.result_label.configure(
                text=f"Detected Season: {season}\n\n{description}",
                background=color_hex
            )
        else:
            self.result_label.configure(text="Unable to determine skin tone.", background="#CCCCCC")

    def classify_skin_tone(self, r, g, b):
        brightness = (r + g + b) / 3
        warm_score = r - b
        if warm_score > 10:
            undertone = 'Warm'
        elif warm_score < -10:
            undertone = 'Cool'
        else:
            undertone = 'Neutral'

        if brightness > 180:
            depth = 'Light'
        elif brightness < 80:
            depth = 'Deep'
        else:
            depth = 'Soft'

        for season, traits in season_types.items():
            if undertone in traits and depth in traits:
                return season, season_tips[season]

        return None


if __name__ == "__main__":
    root = tb.Window(themename="superhero")  # Modern dark theme
    app = FlawntApp(root)
    root.mainloop()
