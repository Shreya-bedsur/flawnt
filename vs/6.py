import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import os
import threading
import time

class ColorAnalyzer:
    def __init__(self):
        self.season_types = {
            'Deep Autumn': {
                'desc': 'Your rich and bold features give you a striking, intense look with natural depth.',
                'undertone': 'Warm',
                'undertone_desc': 'Your skin has warm undertones, reflecting golden, peachy, or yellow hues in natural light.',
                'palette': ['#7B3F00', '#A9746E', '#8B5C2A', '#A0522D', '#6B4226'],
                'cosmetics': {
                    'Foundations': ['Golden Beige', 'Warm Honey'],
                    'Blushes': ['Terracotta', 'Warm Peach'],
                    'Lipsticks': ['Brick Red', 'Burnt Sienna']
                }
            },
            'Warm Autumn': {
                'desc': 'You have a warm, earthy glow that harmonizes with rich, golden colors.',
                'undertone': 'Warm',
                'undertone_desc': 'Your skin leans toward golden, peachy, or yellow undertones.',
                'palette': ['#D2691E', '#CD853F', '#B87333', '#DAA520', '#8B4513'],
                'cosmetics': {
                    'Foundations': ['Honey', 'Golden Tan'],
                    'Blushes': ['Apricot', 'Warm Coral'],
                    'Lipsticks': ['Copper', 'Terracotta']
                }
            },
            'Soft Autumn': {
                'desc': 'You have a muted, gentle look that pairs beautifully with soft, earthy colors.',
                'undertone': 'Warm',
                'undertone_desc': 'Your undertones are warm but subtle, often with a beige or olive tint.',
                'palette': ['#C2B280', '#BC987E', '#D2B48C', '#A67B5B', '#9E7B4F'],
                'cosmetics': {
                    'Foundations': ['Soft Beige', 'Neutral Tan'],
                    'Blushes': ['Muted Peach', 'Soft Coral'],
                    'Lipsticks': ['Rosewood', 'Dusty Coral']
                }
            },
            'Deep Winter': {
                'desc': 'You have a dramatic, high-contrast appearance that shines in cool, bold shades.',
                'undertone': 'Cool',
                'undertone_desc': 'Your skin has cool undertones with hints of blue, pink, or cool beige.',
                'palette': ['#000000', '#4B0082', '#2F4F4F', '#8B0000', '#003366'],
                'cosmetics': {
                    'Foundations': ['Cool Espresso', 'Mocha'],
                    'Blushes': ['Berry', 'Plum'],
                    'Lipsticks': ['Wine', 'Crimson']
                }
            },
            'Cool Winter': {
                'desc': 'Your clear, icy coloring suits bright, high-contrast cool tones.',
                'undertone': 'Cool',
                'undertone_desc': 'Skin tones have blue or rosy undertones, often porcelain or deep ebony.',
                'palette': ['#4169E1', '#00008B', '#9932CC', '#8A2BE2', '#C71585'],
                'cosmetics': {
                    'Foundations': ['Porcelain', 'Cool Almond'],
                    'Blushes': ['Pink Ice', 'Cool Rose'],
                    'Lipsticks': ['Fuchsia', 'Magenta']
                }
            },
            'Bright Winter': {
                'desc': 'You have a striking appearance that thrives in clear, cool, and vivid colors.',
                'undertone': 'Cool',
                'undertone_desc': 'Cool undertones that suit high contrast and jewel tones.',
                'palette': ['#00FFFF', '#00BFFF', '#FF1493', '#9400D3', '#4682B4'],
                'cosmetics': {
                    'Foundations': ['Ivory', 'Cool Beige'],
                    'Blushes': ['Cool Pink', 'Icy Plum'],
                    'Lipsticks': ['Raspberry', 'Cherry Red']
                }
            },
            'Bright Spring': {
                'desc': 'You shine in clear, vivid colors with a fresh and lively brightness.',
                'undertone': 'Warm',
                'undertone_desc': 'Your undertones are warm with a bright, golden glow.',
                'palette': ['#FFD700', '#FFA07A', '#FF69B4', '#00FA9A', '#FF8C00'],
                'cosmetics': {
                    'Foundations': ['Golden Ivory', 'Warm Nude'],
                    'Blushes': ['Coral Pink', 'Soft Peach'],
                    'Lipsticks': ['Poppy Red', 'Bright Coral']
                }
            },
            'Warm Spring': {
                'desc': 'You glow in golden, creamy tones that reflect a sunlit warmth.',
                'undertone': 'Warm',
                'undertone_desc': 'Warm and golden skin undertones, sometimes with peach or apricot hints.',
                'palette': ['#FFDAB9', '#FFA500', '#F0E68C', '#E9967A', '#DAA520'],
                'cosmetics': {
                    'Foundations': ['Warm Beige', 'Peachy Nude'],
                    'Blushes': ['Golden Peach', 'Apricot'],
                    'Lipsticks': ['Coral', 'Warm Rose']
                }
            },
            'Light Spring': {
                'desc': 'You look best in light, warm pastels that are soft and fresh.',
                'undertone': 'Warm',
                'undertone_desc': 'Light warm undertones with peachy, creamy tones.',
                'palette': ['#FFFACD', '#FFE4B5', '#FAD6A5', '#FFDAB9', '#FFB6C1'],
                'cosmetics': {
                    'Foundations': ['Light Warm', 'Soft Ivory'],
                    'Blushes': ['Light Coral', 'Petal Peach'],
                    'Lipsticks': ['Peachy Pink', 'Light Coral']
                }
            },
            'Light Summer': {
                'desc': 'You suit soft, cool pastels and delicate, airy colors.',
                'undertone': 'Cool',
                'undertone_desc': 'Cool undertones with a soft, rosy or bluish tint.',
                'palette': ['#B0C4DE', '#D8BFD8', '#AFEEEE', '#ADD8E6', '#F0F8FF'],
                'cosmetics': {
                    'Foundations': ['Cool Ivory', 'Neutral Porcelain'],
                    'Blushes': ['Cool Rose', 'Light Pink'],
                    'Lipsticks': ['Soft Rose', 'Dusty Pink']
                }
            },
            'Cool Summer': {
                'desc': 'Your soft, elegant appearance pairs well with cool, powdery shades.',
                'undertone': 'Cool',
                'undertone_desc': 'Your skin has cool pink, blue, or neutral undertones.',
                'palette': ['#778899', '#C0C0C0', '#B0E0E6', '#D3D3D3', '#A9A9A9'],
                'cosmetics': {
                    'Foundations': ['Neutral Beige', 'Cool Buff'],
                    'Blushes': ['Cool Mauve', 'Rosy Pink'],
                    'Lipsticks': ['Plum Pink', 'Mauve Rose']
                }
            },
            'Soft Summer': {
                'desc': 'You have a low-contrast look that harmonizes with gentle, muted cool shades.',
                'undertone': 'Cool',
                'undertone_desc': 'Your undertones are soft and cool, sometimes with a hint of neutral beige.',
                'palette': ['#A8B2BD', '#BDBDC6', '#C1A192', '#A9A9A9', '#B6AFA9'],
                'cosmetics': {
                    'Foundations': ['Soft Beige', 'Neutral Cool'],
                    'Blushes': ['Muted Rose', 'Soft Plum'],
                    'Lipsticks': ['Dusty Rose', 'Muted Berry']
                }
            }
        }

    def analyze(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV for better skin detection
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color ranges in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Get skin pixels
        skin_pixels = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        
        # Reshape for clustering
        pixels = skin_pixels.reshape(-1, 3)
        pixels = pixels[pixels.sum(axis=1) > 0]  # Remove black pixels
        
        if len(pixels) == 0:
            return None
        
        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        
        # Get the dominant colors
        colors = kmeans.cluster_centers_.astype(int)
        dominant = colors[0]
        
        # Convert to HSV for better undertone analysis
        h, s, v = colorsys.rgb_to_hsv(dominant[0]/255, dominant[1]/255, dominant[2]/255)
        
        # Determine season based on HSV values
        if v < 0.4:  # Deep
            if h < 0.1:  # Cool
                season = 'Deep Winter'
            else:  # Warm
                season = 'Deep Autumn'
        elif v > 0.7:  # Light
            if h < 0.1:  # Cool
                season = 'Light Summer'
            else:  # Warm
                season = 'Light Spring'
        else:  # Medium
            if h < 0.1:  # Cool
                season = 'Cool Summer'
            else:  # Warm
                season = 'Warm Autumn'
        
        return {
            'season': season,
            'dominant_rgb': tuple(dominant),
            'palette': self.season_types[season]['palette'],
            'desc': self.season_types[season]['desc'],
            'undertone': self.season_types[season]['undertone'],
            'undertone_desc': self.season_types[season]['undertone_desc'],
            'cosmetics': self.season_types[season]['cosmetics']
        }

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, _, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 30
        y = y + cy + self.widget.winfo_rooty() + 30
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief='solid', borderwidth=1, font=("tahoma", 9))
        label.pack(ipadx=4)

    def hide(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class FlawntApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Flawnt Color Analysis')
        self.root.geometry('600x800')
        self.root.minsize(500, 700)
        self.analyzer = ColorAnalyzer()
        self.theme = 'flatly'
        self.history = []
        self.sidebar = None
        self.content = None
        self.camera_running = False
        self.captured_image = None
        self.captured_path = None
        self.analysis = None
        self.create_sidebar()
        self.show_home()

    def create_sidebar(self):
        if self.sidebar:
            self.sidebar.destroy()
        self.sidebar = tb.Frame(self.root, bootstyle='dark')
        self.sidebar.pack(side=LEFT, fill=Y)
        logo = tb.Label(self.sidebar, text='Flawnt', font=('Montserrat', 18, 'bold'), bootstyle='danger')
        logo.pack(pady=(30, 20))
        navs = [
            ('🏠 Home', self.show_home),
            ('🕑 History', self.show_history),
            ('🌗 Theme', self.toggle_theme),
            ('👤 Profile', self.show_profile)
        ]
        for txt, cmd in navs:
            btn = tb.Button(self.sidebar, text=txt, bootstyle='secondary', width=16, command=cmd)
            btn.pack(pady=8)

    def clear_content(self):
        if self.content:
            self.content.destroy()
        self.content = tb.Frame(self.root, bootstyle='light')
        self.content.pack(side=LEFT, fill=BOTH, expand=True)

    def show_home(self):
        self.clear_content()
        card = tb.Frame(self.content, bootstyle='secondary', padding=18)
        card.pack(pady=30, padx=30, fill=X)
        # --- Insert AI/Illustrative Image ---
        try:
            ai_img = Image.open('ai_guide.png').resize((180, 220))
        except FileNotFoundError:
            ai_img = Image.new('RGB', (180, 220), '#d8b4a6')
        ai_img_tk = ImageTk.PhotoImage(ai_img)
        ai_img_label = tk.Label(card, image=ai_img_tk, bd=0)
        ai_img_label.image = ai_img_tk
        ai_img_label.pack(pady=(0, 10))
        # --- Step-by-step Guidance ---
        steps = tb.Frame(card, bootstyle='light')
        steps.pack(pady=10)
        tb.Label(steps, text='How it works:', font=('Montserrat', 10, 'bold')).pack(anchor=W)
        tb.Label(steps, text='1️⃣ Take a clear selfie or upload a photo.', font=('Montserrat', 9)).pack(anchor=W)
        tb.Label(steps, text='2️⃣ Crop to focus on your face.', font=('Montserrat', 9)).pack(anchor=W)
        tb.Label(steps, text='3️⃣ Get your personalized color palette!', font=('Montserrat', 9)).pack(anchor=W)
        # --- Palette Preview (as before) ---
        palette_frame = tb.Frame(card)
        palette_frame.pack(pady=8)
        for color in ['#7B3F00', '#A9746E', '#8B5C2A', '#A0522D', '#6B4226']:
            swatch = tk.Label(palette_frame, width=4, height=2, bg=color, relief='ridge', bd=2)
            swatch.pack(side=LEFT, padx=2)
            ToolTip(swatch, color)
        tb.Label(card, text='Dark Autumn Palette', font=('Montserrat', 10, 'bold')).pack(pady=(8, 0))
        # --- Open Camera Button ---
        tb.Button(self.content, text='Open Camera', bootstyle='danger', width=20, command=self.show_camera).pack(pady=24)
        # --- Selfie Tips (as before) ---
        tips = tb.Frame(self.content, bootstyle='light', padding=10)
        tips.pack(pady=10, fill=X, padx=20)
        tb.Label(tips, text='Selfie tips', font=('Montserrat', 10, 'bold'), bootstyle='secondary').pack(anchor=W)
        tb.Label(tips, text='📷 Please stand in front of good natural light.', font=('Montserrat', 9)).pack(anchor=W)
        tb.Label(tips, text='👓 Take off your glasses.', font=('Montserrat', 9)).pack(anchor=W)

    def show_camera(self):
        self.clear_content()
        self.camera_running = True
        cam_frame = tb.Frame(self.content, bootstyle='secondary', padding=18)
        cam_frame.pack(pady=30, padx=30)
        self.cam_label = tk.Label(cam_frame, width=320, height=240, bd=2, relief='ridge')
        self.cam_label.pack()
        btns = tb.Frame(self.content, bootstyle='light')
        btns.pack(pady=10)
        tb.Button(btns, text='Capture', bootstyle='danger', width=12, command=self.capture_photo).pack(side=LEFT, padx=8)
        tb.Button(btns, text='Upload', bootstyle='secondary', width=12, command=self.upload_photo).pack(side=LEFT, padx=8)
        self.update_camera()

    def update_camera(self):
        if not self.camera_running:
            return
        cap = getattr(self, 'cap', None)
        if cap is None:
            self.cap = cv2.VideoCapture(0)
            cap = self.cap
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((320, 240))
            img_tk = ImageTk.PhotoImage(img)
            self.cam_label.imgtk = img_tk
            self.cam_label.config(image=img_tk)
        self.root.after(30, self.update_camera)

    def capture_photo(self):
        if not hasattr(self, 'cap') or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.camera_running = False
            self.cap.release()
            self.cap = None
            temp_path = 'captured_photo.jpg'
            cv2.imwrite(temp_path, frame)
            self.captured_path = temp_path
            self.show_crop_dialog(temp_path)

    def upload_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg *.jpeg *.png')])
        if file_path:
            self.camera_running = False
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
            self.captured_path = file_path
            self.show_crop_dialog(file_path)

    def show_crop_dialog(self, img_path):
        crop_win = tk.Toplevel(self.root)
        crop_win.title('Crop Image')
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        img = img.resize((320, 320))
        img_tk = ImageTk.PhotoImage(img)
        canvas = tk.Canvas(crop_win, width=320, height=320)
        canvas.pack()
        canvas.create_image(0, 0, anchor='nw', image=img_tk)
        canvas.imgtk = img_tk
        # Simple square crop selector
        rect = canvas.create_rectangle(60, 60, 260, 260, outline='red', width=2)
        def crop_and_continue():
            x1, y1, x2, y2 = canvas.coords(rect)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped = img.crop((x1, y1, x2, y2)).resize((220, 260))
            cropped_path = 'cropped_photo.jpg'
            cropped.save(cropped_path)
            self.captured_path = cropped_path
            crop_win.destroy()
            self.show_preview()
        tb.Button(crop_win, text='Crop & Continue', bootstyle='danger', command=crop_and_continue).pack(pady=8)

    def show_preview(self):
        self.clear_content()
        card = tb.Frame(self.content, bootstyle='secondary', padding=18)
        card.pack(pady=30, padx=30, fill=X)
        img = Image.open(self.captured_path)
        img = img.resize((220, 260))
        img_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(card, image=img_tk, bd=2, relief='ridge')
        img_label.image = img_tk
        img_label.pack()
        tb.Button(self.content, text='Analyze', bootstyle='danger', width=20, command=self.analyze_image).pack(pady=24)

    def analyze_image(self):
        self.clear_content()
        spinner = ttk.Progressbar(self.content, mode='indeterminate', length=200)
        spinner.pack(pady=60)
        tb.Label(self.content, text='Analyzing...', font=('Montserrat', 14, 'bold'), bootstyle='danger').pack(pady=10)
        spinner.start(10)
        threading.Thread(target=self.run_analysis, args=(spinner,), daemon=True).start()

    def run_analysis(self, spinner):
        time.sleep(1.2)  # Simulate processing
        self.analysis = self.analyzer.analyze(self.captured_path)
        self.history.append((self.captured_path, self.analysis))
        self.root.after(100, lambda: self.show_results(spinner))

    def show_results(self, spinner):
        spinner.stop()
        self.clear_content()
        card = tb.Frame(self.content, bootstyle='secondary', padding=18)
        card.pack(pady=30, padx=30, fill=X)
        # Icon
        icon = tk.Label(card, text='🎨', font=('Arial', 32))
        icon.pack()
        # Palette
        palette_frame = tb.Frame(card)
        palette_frame.pack(pady=8)
        for color in self.analysis['palette']:
            swatch = tk.Label(palette_frame, width=4, height=2, bg=color, relief='ridge', bd=2)
            swatch.pack(side=LEFT, padx=2)
            ToolTip(swatch, color)
        # Season/Undertone
        tb.Label(card, text=f"{self.analysis['season']}", font=('Montserrat', 14, 'bold')).pack(pady=(10, 0))
        tb.Label(card, text=self.analysis['desc'], font=('Montserrat', 9), wraplength=300).pack(pady=(2, 0))
        undertone_card = tb.Frame(self.content, bootstyle='light', padding=10)
        undertone_card.pack(pady=10, padx=18, fill=X)
        tb.Label(undertone_card, text='Undertone', font=('Montserrat', 9, 'bold'), bootstyle='danger').pack(anchor=W)
        tb.Label(undertone_card, text=self.analysis['undertone'], font=('Montserrat', 11, 'bold')).pack(anchor=W)
        tb.Label(undertone_card, text=self.analysis['undertone_desc'], font=('Montserrat', 9), wraplength=300).pack(anchor=W)
        # Cosmetics
        rec_frame = tb.Frame(self.content, bootstyle='light', padding=10)
        rec_frame.pack(pady=10, padx=18, fill=X)
        tb.Label(rec_frame, text='Recommended cosmetics', font=('Montserrat', 10, 'bold'), bootstyle='secondary').pack(anchor=W)
        for cat in ['Foundations', 'Blushes', 'Lipsticks']:
            box = tb.Frame(rec_frame, bootstyle='secondary', padding=6)
            box.pack(fill=X, pady=4)
            tb.Label(box, text=cat, font=('Montserrat', 9, 'bold')).pack(anchor=W)
            for item in self.analysis['cosmetics'][cat]:
                tb.Label(box, text=item, font=('Montserrat', 9)).pack(anchor=W)
        # Export
        tb.Button(self.content, text='Export as Image', bootstyle='info', command=self.export_result).pack(pady=10)

    def show_history(self):
        self.clear_content()
        tb.Label(self.content, text='Analysis History', font=('Montserrat', 16, 'bold'), bootstyle='danger').pack(pady=20)
        for idx, (img_path, analysis) in enumerate(reversed(self.history[-10:])):
            frame = tb.Frame(self.content, bootstyle='secondary', padding=8)
            frame.pack(pady=8, padx=20, fill=X)
            img = Image.open(img_path).resize((60, 70))
            img_tk = ImageTk.PhotoImage(img)
            img_label = tk.Label(frame, image=img_tk, bd=1, relief='ridge')
            img_label.image = img_tk
            img_label.pack(side=LEFT, padx=8)
            info = tb.Frame(frame, bootstyle='secondary')
            info.pack(side=LEFT, padx=8)
            tb.Label(info, text=analysis['season'], font=('Montserrat', 10, 'bold')).pack(anchor=W)
            tb.Label(info, text=analysis['undertone'], font=('Montserrat', 9)).pack(anchor=W)
            tb.Label(info, text=analysis['palette'][0], font=('Montserrat', 9)).pack(anchor=W)
            tb.Button(frame, text='View', bootstyle='info', width=8, command=lambda a=analysis: self.show_history_result(a)).pack(side=RIGHT, padx=8)

    def show_history_result(self, analysis):
        self.analysis = analysis
        self.show_results(spinner=None)

    def show_profile(self):
        self.clear_content()
        tb.Label(self.content, text='Profile (Coming Soon)', font=('Montserrat', 16, 'bold'), bootstyle='danger').pack(pady=40)

    def toggle_theme(self):
        self.theme = 'darkly' if self.theme == 'flatly' else 'flatly'
        self.root.style.theme_use(self.theme)

    def export_result(self):
        if not self.analysis:
            return
        file_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG Image', '*.png')])
        if file_path:
            # Create a simple export image
            img = Image.new('RGB', (400, 300), 'white')
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), f"Season: {self.analysis['season']}", fill='black')
            draw.text((20, 50), f"Undertone: {self.analysis['undertone']}", fill='black')
            y = 90
            for color in self.analysis['palette']:
                draw.rectangle([20, y, 80, y+30], fill=color)
                draw.text((90, y+5), color, fill='black')
                y += 40
            img.save(file_path)

if __name__ == '__main__':
    root = tb.Window(themename='flatly')
    app = FlawntApp(root)
    root.mainloop() 