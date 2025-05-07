import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime

class ColorAnalyzer:
    def __init__(self):
        # Initialize color palettes and analysis parameters
        self.seasonal_palettes = {
            'Spring': {
                'description': 'Warm and clear colors with yellow undertones',
                'characteristics': ['Warm undertones', 'Clear and bright complexion', 'Often has freckles'],
                'colors': ['#FFD700', '#FFA07A', '#98FB98', '#87CEEB', '#FFC0CB', '#DDA0DD', '#FFE4B5', '#FAEBD7']
            },
            'Summer': {
                'description': 'Cool and soft colors with blue undertones',
                'characteristics': ['Cool undertones', 'Soft and muted complexion', 'Often has blue veins'],
                'colors': ['#B0E0E6', '#D8BFD8', '#E6E6FA', '#F0F8FF', '#F5F5F5', '#E0FFFF', '#B0C4DE', '#C0C0C0']
            },
            'Autumn': {
                'description': 'Warm and muted colors with golden undertones',
                'characteristics': ['Warm undertones', 'Rich and muted complexion', 'Often has golden highlights'],
                'colors': ['#8B4513', '#A0522D', '#6B8E23', '#556B2F', '#8B6914', '#CD853F', '#DEB887', '#D2B48C']
            },
            'Winter': {
                'description': 'Cool and clear colors with blue undertones',
                'characteristics': ['Cool undertones', 'Clear and bright complexion', 'High contrast features'],
                'colors': ['#000000', '#1B1B1B', '#2C2C2C', '#3D3D3D', '#4E4E4E', '#000080', '#4B0082', '#800080']
            }
        }

        # Enhanced skin tone ranges
        self.skin_tone_ranges = {
            'depth': {
                'Very Deep': (0, 15),
                'Deep': (15, 30),
                'Medium-Deep': (30, 45),
                'Medium': (45, 60),
                'Medium-Light': (60, 75),
                'Light': (75, 85),
                'Very Light': (85, 100)
            },
            'warmth': {
                'Warm': [(0, 30), (330, 360)],
                'Neutral-Warm': [(30, 60), (300, 330)],
                'Neutral': [(60, 90), (270, 300)],
                'Neutral-Cool': [(90, 120), (240, 270)],
                'Cool': [(120, 150), (210, 240)]
            },
            'clarity': {
                'Very Muted': (0, 15),
                'Muted': (15, 30),
                'Soft': (30, 45),
                'Clear': (45, 60),
                'Very Clear': (60, 100)
            }
        }

    def detect_skin_tone(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV for better skin detection
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Enhanced skin color ranges in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin
        skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to get skin pixels
        skin_pixels = cv2.bitwise_and(image_rgb, image_rgb, mask=skin_mask)
        
        # Reshape for k-means
        pixels = skin_pixels.reshape(-1, 3)
        pixels = pixels[pixels.sum(axis=1) > 0]  # Remove black pixels
        
        if len(pixels) == 0:
            raise ValueError("No skin detected in the image")
        
        # Apply k-means clustering with more clusters for better accuracy
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Get the most common color (dominant skin tone)
        dominant_color = colors[np.argmax(np.bincount(labels))]
        
        # Get secondary colors for undertone analysis
        color_counts = np.bincount(labels)
        sorted_indices = np.argsort(color_counts)[::-1]
        secondary_colors = colors[sorted_indices[1:3]]  # Get second and third most common colors
        
        return dominant_color, secondary_colors

    def analyze_skin_tone(self, colors):
        dominant_color, secondary_colors = colors
        
        # Convert RGB to HSV for dominant color
        h, s, v = colorsys.rgb_to_hsv(dominant_color[0]/255, dominant_color[1]/255, dominant_color[2]/255)
        h = h * 360  # Convert hue to degrees
        
        # Analyze secondary colors for undertone
        secondary_hues = []
        for color in secondary_colors:
            h_sec, _, _ = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
            secondary_hues.append(h_sec * 360)
        
        # Determine depth
        depth = v * 100
        depth_category = next((k for k, (min_val, max_val) in self.skin_tone_ranges['depth'].items()
                             if min_val <= depth <= max_val), 'Medium')
        
        # Determine warmth using both dominant and secondary colors
        avg_hue = (h + sum(secondary_hues)) / (len(secondary_hues) + 1)
        warmth_category = next((k for k, ranges in self.skin_tone_ranges['warmth'].items()
                              if any(min_val <= avg_hue <= max_val for min_val, max_val in ranges)), 'Neutral')
        
        # Determine clarity
        clarity = s * 100
        clarity_category = next((k for k, (min_val, max_val) in self.skin_tone_ranges['clarity'].items()
                               if min_val <= clarity <= max_val), 'Soft')
        
        # Calculate undertone intensity
        undertone_intensity = abs(h - 180) / 180  # Distance from neutral (180 degrees)
        
        return {
            'depth': depth_category,
            'warmth': warmth_category,
            'clarity': clarity_category,
            'undertone_intensity': undertone_intensity,
            'hsv': (h, s * 100, v * 100),
            'secondary_hues': secondary_hues
        }

    def determine_season(self, skin_analysis):
        depth = skin_analysis['depth']
        warmth = skin_analysis['warmth']
        clarity = skin_analysis['clarity']
        undertone_intensity = skin_analysis['undertone_intensity']
        
        # Enhanced season determination logic
        if warmth in ['Warm', 'Neutral-Warm']:
            if clarity in ['Clear', 'Very Clear'] and undertone_intensity > 0.3:
                return 'Spring'
            elif clarity in ['Muted', 'Soft'] or undertone_intensity <= 0.3:
                return 'Autumn'
        elif warmth in ['Cool', 'Neutral-Cool']:
            if clarity in ['Clear', 'Very Clear'] and undertone_intensity > 0.3:
                return 'Winter'
            elif clarity in ['Muted', 'Soft'] or undertone_intensity <= 0.3:
                return 'Summer'
        else:  # Neutral
            if depth in ['Very Deep', 'Deep', 'Medium-Deep']:
                return 'Winter'
            elif depth in ['Very Light', 'Light', 'Medium-Light']:
                return 'Summer'
            else:
                return 'Autumn' if undertone_intensity > 0.3 else 'Summer'

    def analyze_image(self, image_path):
        try:
            # Detect skin tone with enhanced analysis
            skin_colors = self.detect_skin_tone(image_path)
            
            # Analyze skin tone with enhanced parameters
            skin_analysis = self.analyze_skin_tone(skin_colors)
            
            # Determine season with enhanced logic
            season = self.determine_season(skin_analysis)
            
            return {
                'skin_tone': skin_colors[0],
                'skin_analysis': skin_analysis,
                'season': season,
                'season_info': self.seasonal_palettes[season],
                'recommended_colors': self.seasonal_palettes[season]['colors']
            }
        except Exception as e:
            return {'error': str(e)}

class ColorAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Color Analysis System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        # Initialize variables
        self.image_path = None
        self.analyzer = ColorAnalyzer()
        self.history = []

        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create left panel for image and controls
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Create right panel for results
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.setup_ui()
        self.setup_menu()

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.browse_image)
        file_menu.add_command(label="Save Analysis", command=self.save_analysis)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="History", command=self.show_history)
        view_menu.add_command(label="Clear History", command=self.clear_history)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Instructions", command=self.show_instructions)

    def setup_ui(self):
        # Image display area
        self.image_frame = ttk.LabelFrame(self.left_panel, text="Image Preview")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image controls
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Image Controls")
        self.control_frame.pack(fill=tk.X, pady=(0, 10))

        # Brightness control
        ttk.Label(self.control_frame, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_scale = ttk.Scale(
            self.control_frame,
            from_=0.0,
            to=2.0,
            orient=tk.HORIZONTAL,
            command=self.adjust_image
        )
        self.brightness_scale.set(1.0)
        self.brightness_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Control buttons
        self.button_frame = ttk.Frame(self.left_panel)
        self.button_frame.pack(fill=tk.X, pady=(0, 10))

        self.browse_button = ttk.Button(
            self.button_frame,
            text="Browse Image",
            command=self.browse_image
        )
        self.browse_button.pack(side=tk.LEFT, padx=5)

        self.analyze_button = ttk.Button(
            self.button_frame,
            text="Analyze Colors",
            command=self.analyze_image,
            state=tk.DISABLED
        )
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(
            self.button_frame,
            text="Reset Image",
            command=self.reset_image,
            state=tk.DISABLED
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Results area
        self.results_frame = ttk.LabelFrame(self.right_panel, text="Analysis Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabbed results
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.overview_tab = ttk.Frame(self.notebook)
        self.colors_tab = ttk.Frame(self.notebook)
        self.palette_tab = ttk.Frame(self.notebook)
        self.details_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.overview_tab, text="Overview")
        self.notebook.add(self.colors_tab, text="Color Analysis")
        self.notebook.add(self.palette_tab, text="Color Palette")
        self.notebook.add(self.details_tab, text="Detailed Analysis")

        # Setup tabs
        self.setup_overview_tab()
        self.setup_colors_tab()
        self.setup_palette_tab()
        self.setup_details_tab()

    def setup_overview_tab(self):
        # Create text widget for overview
        self.overview_text = tk.Text(self.overview_tab, wrap=tk.WORD, height=20)
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.overview_text.config(state=tk.DISABLED)

    def setup_colors_tab(self):
        # Create frame for color analysis
        self.colors_frame = ttk.Frame(self.colors_tab)
        self.colors_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create matplotlib figure for color analysis
        self.colors_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.colors_canvas = FigureCanvasTkAgg(self.colors_fig, master=self.colors_frame)
        self.colors_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_palette_tab(self):
        # Create frame for color palette
        self.palette_frame = ttk.Frame(self.palette_tab)
        self.palette_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create matplotlib figure for color palette
        self.palette_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.palette_canvas = FigureCanvasTkAgg(self.palette_fig, master=self.palette_frame)
        self.palette_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_details_tab(self):
        # Create frame for detailed analysis
        self.details_frame = ttk.Frame(self.details_tab)
        self.details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create text widget for detailed analysis
        self.details_text = tk.Text(self.details_frame, wrap=tk.WORD, height=20)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.config(state=tk.DISABLED)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.image_path = file_path
            self.display_image()
            self.analyze_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)

    def display_image(self):
        if self.image_path:
            # Load and resize image
            self.original_image = Image.open(self.image_path)
            self.current_image = self.original_image.copy()
            self.update_image_display()

    def update_image_display(self):
        if hasattr(self, 'current_image'):
            # Resize while maintaining aspect ratio
            display_image = self.current_image.copy()
            display_image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(display_image)
            
            # Update image label
            self.image_label.configure(image=photo)
            self.image_label.image = photo

    def adjust_image(self, value):
        if hasattr(self, 'original_image'):
            # Apply brightness adjustment
            enhancer = ImageEnhance.Brightness(self.original_image)
            self.current_image = enhancer.enhance(float(value))
            self.update_image_display()

    def reset_image(self):
        if hasattr(self, 'original_image'):
            self.current_image = self.original_image.copy()
            self.brightness_scale.set(1.0)
            self.update_image_display()

    def analyze_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first")
            return

        try:
            # Save current image state
            temp_path = "temp_analysis.jpg"
            self.current_image.save(temp_path)

            # Perform analysis
            results = self.analyzer.analyze_image(temp_path)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            if 'error' in results:
                messagebox.showerror("Error", results['error'])
                return

            # Add to history
            self.history.append({
                'timestamp': datetime.now(),
                'image_path': self.image_path,
                'results': results
            })

            # Update all tabs
            self.update_overview(results)
            self.update_colors_analysis(results)
            self.update_color_palette(results)
            self.update_detailed_analysis(results)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def update_overview(self, results):
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete(1.0, tk.END)
        
        overview_text = f"""
Seasonal Color Analysis Results
==============================

Season: {results['season']}

Description:
{results['season_info']['description']}

Characteristics:
{chr(10).join(f'- {char}' for char in results['season_info']['characteristics'])}

Skin Tone Analysis:
- Depth: {results['skin_analysis']['depth']}
- Warmth: {results['skin_analysis']['warmth']}
- Clarity: {results['skin_analysis']['clarity']}
- Undertone Intensity: {results['skin_analysis']['undertone_intensity']:.2f}

HSV Analysis:
- Hue: {results['skin_analysis']['hsv'][0]:.1f}°
- Saturation: {results['skin_analysis']['hsv'][1]:.1f}%
- Value: {results['skin_analysis']['hsv'][2]:.1f}%
"""
        self.overview_text.insert(tk.END, overview_text)
        self.overview_text.config(state=tk.DISABLED)

    def update_colors_analysis(self, results):
        self.colors_fig.clear()
        ax = self.colors_fig.add_subplot(111, projection='polar')
        
        # Plot color wheel
        theta = np.linspace(0, 2*np.pi, 360)
        r = np.ones_like(theta)
        ax.plot(theta, r, color='gray', alpha=0.3)
        
        # Plot skin tone position
        h = results['skin_analysis']['hsv'][0] * np.pi/180
        ax.scatter(h, 1, color='red', s=100, label='Skin Tone')
        
        # Plot secondary colors
        for i, h_sec in enumerate(results['skin_analysis']['secondary_hues']):
            ax.scatter(h_sec * np.pi/180, 0.8, color='blue', s=50, label=f'Secondary {i+1}')
        
        # Add color regions
        regions = {
            'Warm': (0, 60),
            'Neutral-Warm': (60, 120),
            'Neutral': (120, 180),
            'Neutral-Cool': (180, 240),
            'Cool': (240, 300)
        }
        
        for region, (start, end) in regions.items():
            ax.fill_between(
                np.linspace(start * np.pi/180, end * np.pi/180, 100),
                0, 1,
                alpha=0.1,
                label=region
            )
        
        ax.set_title('Color Wheel Position')
        ax.legend(bbox_to_anchor=(1.1, 1.1))
        self.colors_canvas.draw()

    def update_color_palette(self, results):
        self.palette_fig.clear()
        ax = self.palette_fig.add_subplot(111)
        
        # Plot color palette
        colors = results['recommended_colors']
        for i, color in enumerate(colors):
            rect = plt.Rectangle((i/len(colors), 0), 1/len(colors), 1, color=color)
            ax.add_patch(rect)
            # Add color hex code
            ax.text(
                (i + 0.5)/len(colors),
                0.5,
                color,
                color='white' if self.is_dark_color(color) else 'black',
                ha='center',
                va='center'
            )
        
        ax.set_title('Recommended Color Palette')
        ax.axis('off')
        self.palette_canvas.draw()

    def update_detailed_analysis(self, results):
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        details_text = f"""
Detailed Color Analysis
======================

Skin Tone Analysis:
------------------
Depth Category: {results['skin_analysis']['depth']}
- This indicates the overall lightness/darkness of your skin tone
- Affects how colors appear against your skin

Warmth Category: {results['skin_analysis']['warmth']}
- Indicates whether your skin has warm (yellow/golden) or cool (pink/blue) undertones
- Helps determine which color temperatures complement your skin

Clarity Category: {results['skin_analysis']['clarity']}
- Describes how clear or muted your skin tone appears
- Influences which color intensities work best for you

Undertone Intensity: {results['skin_analysis']['undertone_intensity']:.2f}
- Measures how strong your undertone is
- Higher values indicate more pronounced undertones

HSV Color Space Analysis:
------------------------
Hue: {results['skin_analysis']['hsv'][0]:.1f}°
- Represents the base color of your skin tone
- Measured in degrees around the color wheel

Saturation: {results['skin_analysis']['hsv'][1]:.1f}%
- Indicates the intensity of your skin color
- Higher values mean more vibrant colors

Value: {results['skin_analysis']['hsv'][2]:.1f}%
- Represents the brightness of your skin tone
- Affects how light or dark colors appear on you

Seasonal Color Analysis:
-----------------------
Season: {results['season']}
Description: {results['season_info']['description']}

Key Characteristics:
{chr(10).join(f'- {char}' for char in results['season_info']['characteristics'])}

Color Recommendations:
--------------------
The following colors are recommended for your color season:
{chr(10).join(f'- {color}' for color in results['recommended_colors'])}

These colors are chosen to complement your natural coloring and enhance your features.
"""
        self.details_text.insert(tk.END, details_text)
        self.details_text.config(state=tk.DISABLED)

    def is_dark_color(self, hex_color):
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Calculate perceived brightness
        brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        return brightness < 0.5

    def save_analysis(self):
        if not hasattr(self, 'current_image'):
            messagebox.showerror("Error", "No analysis to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.overview_text.get(1.0, tk.END))
                messagebox.showinfo("Success", "Analysis saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save analysis: {str(e)}")

    def show_history(self):
        if not self.history:
            messagebox.showinfo("History", "No analysis history available")
            return

        history_window = tk.Toplevel(self.root)
        history_window.title("Analysis History")
        history_window.geometry("600x400")

        # Create text widget for history
        history_text = tk.Text(history_window, wrap=tk.WORD)
        history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add history entries
        for entry in reversed(self.history):
            history_text.insert(tk.END, f"""
Analysis from {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Image: {os.path.basename(entry['image_path'])}
Season: {entry['results']['season']}
Skin Tone: {entry['results']['skin_analysis']['depth']} ({entry['results']['skin_analysis']['warmth']})
Undertone Intensity: {entry['results']['skin_analysis']['undertone_intensity']:.2f}
----------------------------------------
""")

        history_text.config(state=tk.DISABLED)

    def clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear the analysis history?"):
            self.history = []
            messagebox.showinfo("Success", "History cleared successfully")

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Advanced Color Analysis System\n\n"
            "Version 2.0\n"
            "A comprehensive tool for analyzing skin tones and providing color recommendations.\n\n"
            "Features:\n"
            "- Enhanced skin tone detection and analysis\n"
            "- Detailed undertone analysis\n"
            "- Seasonal color analysis\n"
            "- Color palette recommendations\n"
            "- Detailed color space analysis\n"
            "- Image adjustment capabilities\n"
            "- Analysis history tracking"
        )

    def show_instructions(self):
        instructions = """
How to Use the Color Analysis System:

1. Loading an Image:
   - Click 'Browse Image' to select a photo
   - The image will be displayed in the preview area
   - Use the brightness slider to adjust the image if needed

2. Analyzing Colors:
   - Click 'Analyze Colors' to perform the analysis
   - Results will be displayed in four tabs:
     * Overview: Summary of the analysis
     * Color Analysis: Color wheel visualization
     * Color Palette: Recommended colors
     * Detailed Analysis: In-depth color information

3. Saving Results:
   - Use File > Save Analysis to save the results
   - The analysis will be saved as a text file

4. Viewing History:
   - Use View > History to see previous analyses
   - Clear history using View > Clear History

Tips for Best Results:
- Use well-lit photos with clear skin visibility
- Avoid heavy makeup or filters
- Ensure the photo shows natural skin tone
- Use the brightness adjustment if needed
"""
        messagebox.showinfo("Instructions", instructions)

def main():
    root = tk.Tk()
    app = ColorAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

    
    