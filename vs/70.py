import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import os
import threading

# --- Color Analysis Logic (improved) ---
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
            'Light Spring': {
                'desc': 'Fresh, light, and warm features with a radiant glow.',
                'undertone': 'Warm',
                'undertone_desc': 'Your skin has warm undertones, reflecting peachy or golden hues.',
                'palette': ['#FFD8B1', '#FFE4B5', '#F5DEB3', '#FFDAB9', '#FFB347'],
                'cosmetics': {
                    'Foundations': ['Ivory', 'Light Beige'],
                    'Blushes': ['Peach', 'Coral'],
                    'Lipsticks': ['Peach', 'Coral Pink']
                }
            },
            'Cool Winter': {
                'desc': 'High contrast, cool, and clear features with blue or pink undertones.',
                'undertone': 'Cool',
                'undertone_desc': 'Your skin has cool undertones, reflecting pink or blue hues.',
                'palette': ['#2C2C54', '#474787', '#AAABB8', '#E84545', '#903749'],
                'cosmetics': {
                    'Foundations': ['Porcelain', 'Cool Beige'],
                    'Blushes': ['Rose', 'Berry'],
                    'Lipsticks': ['Fuchsia', 'Berry Red']
                }
            }
        }

    def analyze(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        skin_pixels = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        pixels = skin_pixels.reshape(-1, 3)
        pixels = pixels[pixels.sum(axis=1) > 0]
        if len(pixels) == 0:
            return None
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        dominant = colors[0]
        h, s, v = colorsys.rgb_to_hsv(dominant[0]/255, dominant[1]/255, dominant[2]/255)
        if v < 0.4:
            season = 'Deep Autumn'
        elif h > 0.1 and h < 0.2:
            season = 'Light Spring'
        else:
            season = 'Cool Winter'
        return {
            'season': season,
            'dominant_rgb': tuple(dominant),
            'palette': self.season_types[season]['palette'],
            'desc': self.season_types[season]['desc'],
            'undertone': self.season_types[season]['undertone'],
            'undertone_desc': self.season_types[season]['undertone_desc'],
            'cosmetics': self.season_types[season]['cosmetics']
        }

# --- UI ---
class FlawntApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Flawnt Color Analysis')
        self.root.geometry('420x800')
        self.root.resizable(False, False)
        self.analyzer = ColorAnalyzer()
        self.image_path = None
        self.analysis = None
        self.style = tb.Style('flatly')
        self.create_home()

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_home(self):
        self.clear()
        frame = tb.Frame(self.root, bootstyle='light')
        frame.pack(fill=BOTH, expand=True)
        logo = tb.Label(frame, text='Flawnt', font=('Montserrat', 18, 'bold'), bootstyle='danger')
        logo.pack(pady=(20, 0))
        card = tb.Frame(frame, bootstyle='secondary', padding=10)
        card.pack(pady=20)
        img = Image.open('demo_face.jpg') if os.path.exists('demo_face.jpg') else Image.new('RGB', (180, 220), '#d8b4a6')
        img = img.resize((180, 220))
        img_tk = ImageTk.PhotoImage(img)
        img_label = tb.Label(card, image=img_tk)
        img_label.image = img_tk
        img_label.pack()
        palette_frame = tb.Frame(card)
        palette_frame.pack(pady=8)
        for idx, color in enumerate(['#7B3F00', '#A9746E', '#8B5C2A', '#A0522D', '#6B4226']):
         style_name = f"Swatch{idx}.TLabel"
         self.style.configure(style_name, background=color)
         swatch = tb.Label(palette_frame, width=3, style=style_name)
         swatch.pack(side=LEFT, padx=2)

        tb.Label(card, text='Dark Autumn Palette', font=('Montserrat', 10, 'bold')).pack(pady=(8, 0))
        tb.Button(frame, text='Open Camera', bootstyle='danger', width=20, command=self.open_camera).pack(pady=18)
        tips = tb.Frame(frame, bootstyle='light', padding=10)
        tips.pack(pady=10, fill=X, padx=20)
        tb.Label(tips, text='Selfie tips', font=('Montserrat', 10, 'bold'), bootstyle='secondary').pack(anchor=W)
        tb.Label(tips, text='📷 Please stand in front of good natural light.', font=('Montserrat', 9)).pack(anchor=W)
        tb.Label(tips, text='👓 Take off your glasses.', font=('Montserrat', 9)).pack(anchor=W)
        self.create_navbar(frame, 0)

    def open_camera(self):
        self.clear()
        frame = tb.Frame(self.root, bootstyle='light')
        frame.pack(fill=BOTH, expand=True)
        tb.Label(frame, text='Image capture', font=('Montserrat', 14, 'bold')).pack(pady=18)
        self.img_preview = tk.Label(frame, bg='#e0e0e0', width=220, height=260)

        self.img_preview.pack(pady=10)
        btn_frame = tb.Frame(frame, bootstyle='light')
        btn_frame.pack(pady=10)
        tb.Button(btn_frame, text='Retake', bootstyle='secondary', width=12, command=self.capture_from_camera).pack(side=LEFT, padx=8)
        tb.Button(btn_frame, text='Confirm', bootstyle='danger', width=12, command=self.confirm_image).pack(side=LEFT, padx=8)
        self.selected_img = None
        self.capture_from_camera()
        self.create_navbar(frame, 1)

    def capture_from_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror('Error', 'Could not open webcam.')
            return
        cv2.namedWindow('Capture Photo (Press SPACE to capture, ESC to cancel)')
        img_captured = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Capture Photo (Press SPACE to capture, ESC to cancel)', frame)
            key = cv2.waitKey(1)
            if key % 256 == 27:  # ESC pressed
                break
            elif key % 256 == 32:  # SPACE pressed
                img_captured = frame
                break
        cap.release()
        cv2.destroyAllWindows()
        if img_captured is not None:
            temp_path = 'captured_photo.jpg'
            cv2.imwrite(temp_path, img_captured)
            self.image_path = temp_path
            img = Image.fromarray(cv2.cvtColor(img_captured, cv2.COLOR_BGR2RGB)).resize((220, 260))
            img_tk = ImageTk.PhotoImage(img)
            self.img_preview.config(image=img_tk)
            self.img_preview.image = img_tk
            self.selected_img = temp_path

    def confirm_image(self):
        if not self.selected_img:
            messagebox.showerror('Error', 'Please capture or select an image.')
            return
        self.show_processing()
        threading.Thread(target=self.run_analysis, daemon=True).start()

    def show_processing(self):
        self.clear()
        frame = tb.Frame(self.root, bootstyle='light')
        frame.pack(fill=BOTH, expand=True)
        tb.Label(frame, text='Analyzing...', font=('Montserrat', 14, 'bold'), bootstyle='danger').pack(pady=40)
        # Show palette placeholder
        palette_frame = tb.Frame(frame, bootstyle='light')
        palette_frame.pack(pady=20)
        for color in ['#e0e0e0']*5:
            swatch = tb.Label(palette_frame, width=4, bootstyle='secondary')
            swatch.config(bg=color)
            swatch.pack(side=LEFT, padx=2)
        self.root.update()

    def run_analysis(self):
        self.analysis = self.analyzer.analyze(self.selected_img)
        self.root.after(100, self.create_recommendation)

    def create_recommendation(self):
        self.clear()
        frame = tb.Frame(self.root, bootstyle='light')
        frame.pack(fill=BOTH, expand=True)
        card = tb.Frame(frame, bootstyle='secondary', padding=12)
        card.pack(pady=18, padx=18, fill=X)
        swatch = tb.Label(card, width=10, height=5)
        swatch.config(bg=self.analysis['palette'][0])
        swatch.pack(side=LEFT, padx=8)
        info = tb.Frame(card, bootstyle='secondary')
        info.pack(side=LEFT, padx=8)
        tb.Label(info, text='Season Type', font=('Montserrat', 9, 'bold'), bootstyle='danger').pack(anchor=W)
        tb.Label(info, text=self.analysis['season'], font=('Montserrat', 12, 'bold')).pack(anchor=W)
        tb.Label(info, text=self.analysis['desc'], font=('Montserrat', 9), wraplength=180).pack(anchor=W, pady=(2, 0))
        undertone_card = tb.Frame(frame, bootstyle='light', padding=10)
        undertone_card.pack(pady=10, padx=18, fill=X)
        tb.Label(undertone_card, text='Undertone', font=('Montserrat', 9, 'bold'), bootstyle='danger').pack(anchor=W)
        tb.Label(undertone_card, text=self.analysis['undertone'], font=('Montserrat', 11, 'bold')).pack(anchor=W)
        tb.Label(undertone_card, text=self.analysis['undertone_desc'], font=('Montserrat', 9), wraplength=300).pack(anchor=W)
        palette_frame = tb.Frame(frame, bootstyle='light')
        palette_frame.pack(pady=8)
        for color in self.analysis['palette']:
            swatch = tb.Label(palette_frame, width=4, bootstyle='secondary')
            swatch.config(bg=color)
            swatch.pack(side=LEFT, padx=2)
        rec_frame = tb.Frame(frame, bootstyle='light', padding=10)
        rec_frame.pack(pady=10, padx=18, fill=X)
        tb.Label(rec_frame, text='Recommended cosmetics', font=('Montserrat', 10, 'bold'), bootstyle='secondary').pack(anchor=W)
        for cat in ['Foundations', 'Blushes', 'Lipsticks']:
            box = tb.Frame(rec_frame, bootstyle='secondary', padding=6)
            box.pack(fill=X, pady=4)
            tb.Label(box, text=cat, font=('Montserrat', 9, 'bold')).pack(anchor=W)
            for item in self.analysis['cosmetics'][cat]:
                tb.Label(box, text=item, font=('Montserrat', 9)).pack(anchor=W)
        self.create_navbar(frame, 4)

    def create_navbar(self, parent, active):
        nav = tb.Frame(parent, bootstyle='light')
        nav.pack(side=BOTTOM, fill=X, pady=8)
        icons = ['🏠', '❓', '💖', '👤']
        labels = ['Home', 'Ask It', 'Wishlist', 'Profile']
        for i, (icon, label) in enumerate(zip(icons, labels)):
            style = 'danger' if i == active else 'secondary'
            btn = tb.Button(nav, text=f'{icon}\n{label}', bootstyle=style, width=8, command=self.create_home if i == 0 else None)
            btn.pack(side=LEFT, padx=2)

if __name__ == '__main__':
    root = tb.Window(themename='flatly')
    app = FlawntApp(root)
    root.mainloop()
