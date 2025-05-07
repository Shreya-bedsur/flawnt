import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import colorsys
from typing import Dict, List, Tuple, Union
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

class EnhancedColorAnalyzer:
    def __init__(self):
        # Seasonal Color Palettes based on Style Coaching Institute
        self.seasonal_palettes = {
            'Spring': {
                'description': 'Warm and clear colors with yellow undertones',
                'characteristics': ['Warm undertones', 'Clear and bright complexion', 'Often has freckles'],
                'colors': [
                    '#FFD700',  # Gold
                    '#FFA07A',  # Light Salmon
                    '#98FB98',  # Pale Green
                    '#87CEEB',  # Sky Blue
                    '#FFC0CB',  # Pink
                    '#DDA0DD',  # Plum
                    '#F0E68C',  # Khaki
                    '#E6E6FA',  # Lavender
                    '#FFE4B5',  # Moccasin
                    '#FAEBD7',  # Antique White
                    '#FFEFD5',  # Papaya Whip
                    '#FFF5E6'   # Seashell
                ]
            },
            'Summer': {
                'description': 'Cool and soft colors with blue undertones',
                'characteristics': ['Cool undertones', 'Soft and muted complexion', 'Often has blue veins'],
                'colors': [
                    '#B0E0E6',  # Powder Blue
                    '#D8BFD8',  # Thistle
                    '#E6E6FA',  # Lavender
                    '#F0F8FF',  # Alice Blue
                    '#F5F5F5',  # White Smoke
                    '#E0FFFF',  # Light Cyan
                    '#F0FFFF',  # Azure
                    '#B0C4DE',  # Light Steel Blue
                    '#C0C0C0',  # Silver
                    '#DCDCDC',  # Gainsboro
                    '#DDA0DD',  # Plum
                    '#DEB887'   # Burlywood
                ]
            },
            'Autumn': {
                'description': 'Warm and muted colors with golden undertones',
                'characteristics': ['Warm undertones', 'Rich and muted complexion', 'Often has golden highlights'],
                'colors': [
                    '#8B4513',  # Saddle Brown
                    '#A0522D',  # Sienna
                    '#6B8E23',  # Olive Drab
                    '#556B2F',  # Dark Olive Green
                    '#8B6914',  # Dark Goldenrod
                    '#CD853F',  # Peru
                    '#DEB887',  # Burlywood
                    '#D2B48C',  # Tan
                    '#BC8F8F',  # Rosy Brown
                    '#F5DEB3',  # Wheat
                    '#DAA520',  # Goldenrod
                    '#B8860B'   # Dark Goldenrod
                ]
            },
            'Winter': {
                'description': 'Cool and clear colors with blue undertones',
                'characteristics': ['Cool undertones', 'Clear and bright complexion', 'High contrast features'],
                'colors': [
                    '#000000',  # Black
                    '#1B1B1B',  # Dark Gray
                    '#2C2C2C',  # Darker Gray
                    '#3D3D3D',  # Medium Gray
                    '#4E4E4E',  # Lighter Gray
                    '#000080',  # Navy
                    '#4B0082',  # Indigo
                    '#800080',  # Purple
                    '#FF0000',  # Red
                    '#00FF00',  # Green
                    '#0000FF',  # Blue
                    '#FFFFFF'   # White
                ]
            }
        }

        # Enhanced skin tone ranges for more accurate detection
        self.skin_tone_ranges = {
            'depth': {
                'Dark': (0, 15),
                'Deep': (15, 30),
                'Tan': (30, 45),
                'Olive': (45, 60),
                'Medium': (60, 75),
                'Light': (75, 85),
                'Fair': (85, 100)
            },
            'warmth': {
                'Warm': [(0, 20), (340, 360)],
                'Neutral-Warm': [(20, 40), (320, 340)],
                'Neutral': [(40, 60), (300, 320)],
                'Neutral-Cool': [(60, 80), (280, 300)],
                'Cool': [(80, 100), (260, 280)]
            },
            'clarity': {
                'Very Muted': (0, 15),
                'Muted': (15, 30),
                'Soft': (30, 45),
                'Clear': (45, 60),
                'Very Clear': (60, 100)
            }
        }

        # Set the style for better visualization using matplotlib's built-in style
        plt.style.use('ggplot')

    def determine_season(self, skin_analysis: Dict) -> str:
        """Determine season based on skin analysis"""
        depth = skin_analysis['depth']
        warmth = skin_analysis['warmth']
        clarity = skin_analysis['clarity']
        
        # Spring types
        if warmth in ['Warm', 'Neutral-Warm'] and clarity in ['Clear', 'Very Clear']:
            return 'Spring'
        
        # Summer types
        elif warmth in ['Cool', 'Neutral-Cool'] and clarity in ['Muted', 'Soft']:
            return 'Summer'
        
        # Autumn types
        elif warmth in ['Warm', 'Neutral-Warm'] and clarity in ['Muted', 'Soft']:
            return 'Autumn'
        
        # Winter types
        else:
            return 'Winter'

    def get_recommended_colors(self, season: str) -> List[str]:
        """Get recommended colors based on seasonal analysis"""
        return self.seasonal_palettes.get(season, self.seasonal_palettes['Spring'])['colors']

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze image with enhanced error handling and validation"""
        try:
            # Open and validate image
            image = Image.open(image_path)
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
            
            # Get dominant color with enhanced detection
            dominant_color = self.get_dominant_colors(image)[0]
            
            # Analyze skin tone with enhanced analysis
            skin_analysis = self.analyze_skin_tone(dominant_color)
            
            # Determine season
            season = self.determine_season(skin_analysis)
            
            # Get recommended colors
            recommended_colors = self.get_recommended_colors(season)
            
            return {
                'skin_tone': dominant_color,
                'skin_analysis': skin_analysis,
                'season': season,
                'recommended_colors': recommended_colors,
                'season_info': self.seasonal_palettes[season]
            }
        except Exception as e:
            return {'error': str(e)}

    def visualize_results(self, image_path: str, results: Dict) -> None:
        """Enhanced visualization with seasonal color analysis"""
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.5])
        
        # Set background color
        fig.patch.set_facecolor('#F0F8FF')
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14, pad=20)
        ax1.axis('off')
        
        # Color palette
        ax2 = fig.add_subplot(gs[0, 1])
        colors = results['recommended_colors']
        for i, color in enumerate(colors):
            ax2.add_patch(Rectangle((i/len(colors), 0), 1/len(colors), 1, color=color))
        ax2.set_title('Recommended Colors', fontsize=14, pad=20)
        ax2.axis('off')
        
        # Seasonal analysis
        ax3 = fig.add_subplot(gs[1, :])
        season_info = results['season_info']
        analysis = results['skin_analysis']
        
        info_text = f"""
        Seasonal Color Analysis:
        Season: {results['season']}
        
        Description:
        {season_info['description']}
        
        Characteristics:
        {chr(10).join(f'- {char}' for char in season_info['characteristics'])}
        
        Skin Tone Analysis:
        - Depth: {analysis['depth']}
        - Warmth: {analysis['warmth']}
        - Clarity: {analysis['clarity']}
        """
        ax3.text(0.1, 0.5, info_text, fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.8))
        ax3.axis('off')
        
        # HSV Analysis
        ax4 = fig.add_subplot(gs[2, :])
        h, s, v = analysis['hsv']
        hsv_text = f"""
        HSV Analysis:
        - Hue: {h:.1f}° (Color wheel position)
        - Saturation: {s:.1f}% (Color intensity)
        - Value: {v:.1f}% (Brightness)
        """
        ax4.text(0.1, 0.5, hsv_text, fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.8))
        ax4.axis('off')
        
        # Color wheel
        ax5 = fig.add_subplot(gs[3, :], projection='polar')
        theta = np.linspace(0, 2*np.pi, 360)
        r = np.ones_like(theta)
        ax5.plot(theta, r, color='gray', alpha=0.3)
        ax5.scatter(h * np.pi/180, 1, color='red', s=100)
        ax5.set_title('Color Wheel Position', fontsize=12, pad=20)
        ax5.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    analyzer = EnhancedColorAnalyzer()
    
    print("Seasonal Color Analysis System")
    print("============================")
    image_path = input("\nEnter the path to your image: ")
    
    print("\nAnalyzing image...")
    results = analyzer.analyze_image(image_path)
    
    if 'error' not in results:
        print("\nAnalysis Results:")
        print("----------------")
        print(f"Season: {results['season']}")
        print(f"\nDescription:")
        print(results['season_info']['description'])
        
        print("\nCharacteristics:")
        for char in results['season_info']['characteristics']:
            print(f"- {char}")
        
        print("\nSkin Tone Analysis:")
        print(f"- Depth: {results['skin_analysis']['depth']}")
        print(f"- Warmth: {results['skin_analysis']['warmth']}")
        print(f"- Clarity: {results['skin_analysis']['clarity']}")
        
        print("\nRecommended Colors:")
        for color in results['recommended_colors']:
            print(f"- {color}")
        
        # Visualize results
        analyzer.visualize_results(image_path, results)
    else:
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()