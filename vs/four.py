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
        # Enhanced 12-Tone Seasonal Color System with expanded palettes
        self.seasonal_palettes = {
            # Spring Types
            'Light Spring': {
                'colors': ['#FFF5E6', '#FFDAB9', '#B0E0E6', '#FFB6C1', '#F5DEB3', '#FFE4B5', '#FAEBD7', '#FFEFD5',
                          '#FFD700', '#FFA07A', '#98FB98', '#87CEEB', '#FFC0CB', '#DDA0DD', '#F0E68C', '#E6E6FA'],
                'description': 'Light, warm, and clear colors - perfect for light skin with golden undertones',
                'characteristics': ['Light to medium skin with golden undertones', 'Often has freckles', 'Warm, peachy complexion'],
                'style': 'Light and fresh with warm undertones'
            },
            'True Spring': {
                'colors': ['#FFD700', '#FF4500', '#FA8072', '#40E0D0', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347',
                          '#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500', '#FF8C00', '#FFA500'],
                'description': 'Medium contrast, warm colors - ideal for medium skin with golden undertones',
                'characteristics': ['Medium skin with golden undertones', 'Clear, warm complexion', 'Natural warmth in skin'],
                'style': 'Warm and vibrant with medium contrast'
            },
            'Clear Spring': {
                'colors': ['#FF69B4', '#32CD32', '#0000CD', '#FF0000', '#FFD700', '#00FF00', '#FF1493', '#00CED1',
                          '#FF69B4', '#32CD32', '#0000CD', '#FF0000', '#FFD700', '#00FF00', '#FF1493', '#00CED1'],
                'description': 'High contrast, warm but bright colors - suits clear, bright complexions',
                'characteristics': ['Clear, bright complexion', 'High contrast features', 'Warm but bright undertones'],
                'style': 'Bright and clear with high contrast'
            },
            
            # Summer Types
            'Light Summer': {
                'colors': ['#E6E6FA', '#98FB98', '#B0E0E6', '#FFB6C1', '#F0F8FF', '#F5F5F5', '#E0FFFF', '#F0FFFF',
                          '#E6E6FA', '#98FB98', '#B0E0E6', '#FFB6C1', '#F0F8FF', '#F5F5F5', '#E0FFFF', '#F0FFFF'],
                'description': 'Light, cool, and soft colors - perfect for light, ashy skin tones',
                'characteristics': ['Light, ashy skin tones', 'Cool undertones', 'Soft, muted features'],
                'style': 'Soft and cool with light tones'
            },
            'True Summer': {
                'colors': ['#DDA0DD', '#DEB887', '#000080', '#A9A9A9', '#B0C4DE', '#C0C0C0', '#D8BFD8', '#DDA0DD',
                          '#DDA0DD', '#DEB887', '#000080', '#A9A9A9', '#B0C4DE', '#C0C0C0', '#D8BFD8', '#DDA0DD'],
                'description': 'Cool and muted colors - ideal for cool undertones with medium contrast',
                'characteristics': ['Medium skin with cool undertones', 'Muted features', 'Balanced contrast'],
                'style': 'Cool and balanced with medium contrast'
            },
            'Soft Summer': {
                'colors': ['#708090', '#FFB6C1', '#E6E6FA', '#B0C4DE', '#D8BFD8', '#DCDCDC', '#E6E6FA', '#F0F8FF',
                          '#708090', '#FFB6C1', '#E6E6FA', '#B0C4DE', '#D8BFD8', '#DCDCDC', '#E6E6FA', '#F0F8FF'],
                'description': 'Cool and very muted colors - suits soft, muted complexions',
                'characteristics': ['Soft, muted complexion', 'Cool undertones', 'Low contrast features'],
                'style': 'Soft and muted with cool undertones'
            },
            
            # Autumn Types
            'Soft Autumn': {
                'colors': ['#A9A9A9', '#9ACD32', '#DEB887', '#6B8E23', '#D2B48C', '#BC8F8F', '#CD853F', '#DEB887',
                          '#A9A9A9', '#9ACD32', '#DEB887', '#6B8E23', '#D2B48C', '#BC8F8F', '#CD853F', '#DEB887'],
                'description': 'Warm and muted colors - perfect for warm, muted complexions',
                'characteristics': ['Warm, muted complexion', 'Golden or olive undertones', 'Soft features'],
                'style': 'Warm and muted with soft features'
            },
            'True Autumn': {
                'colors': ['#FF7F00', '#D2691E', '#DAA520', '#556B2F', '#8B4513', '#CD853F', '#DEB887', '#D2B48C',
                          '#FF7F00', '#D2691E', '#DAA520', '#556B2F', '#8B4513', '#CD853F', '#DEB887', '#D2B48C'],
                'description': 'Warm and rich colors - ideal for golden, tan skin tones',
                'characteristics': ['Golden or tan skin', 'Rich, warm undertones', 'Medium to deep complexion'],
                'style': 'Rich and warm with golden undertones'
            },
            'Deep Autumn': {
                'colors': ['#800020', '#556B2F', '#8B4513', '#D2691E', '#8B0000', '#4B0082', '#800080', '#4B0082',
                          '#800020', '#556B2F', '#8B4513', '#D2691E', '#8B0000', '#4B0082', '#800080', '#4B0082'],
                'description': 'Dark, warm, and muted colors - suits deep, rich complexions',
                'characteristics': ['Deep, rich complexion', 'Warm undertones', 'High contrast features'],
                'style': 'Deep and rich with warm undertones'
            },
            
            # Winter Types
            'Cool Winter': {
                'colors': ['#FF00FF', '#0000CD', '#36454F', '#FFB6C1', '#000000', '#4B0082', '#800080', '#000080',
                          '#FF00FF', '#0000CD', '#36454F', '#FFB6C1', '#000000', '#4B0082', '#800080', '#000080'],
                'description': 'Cool and high contrast colors - perfect for cool undertones with high contrast',
                'characteristics': ['Cool undertones', 'High contrast features', 'Clear, bright complexion'],
                'style': 'Cool and high contrast with clear features'
            },
            'Deep Winter': {
                'colors': ['#800080', '#000000', '#DC143C', '#000080', '#4B0082', '#191970', '#000080', '#000000',
                          '#800080', '#000000', '#DC143C', '#000080', '#4B0082', '#191970', '#000080', '#000000'],
                'description': 'Cool, dark, and clear colors - ideal for deep skin with cool undertones',
                'characteristics': ['Deep skin with cool undertones', 'High contrast', 'Clear features'],
                'style': 'Deep and cool with high contrast'
            },
            'Clear Winter': {
                'colors': ['#FF0000', '#00FFFF', '#FF69B4', '#000000', '#FFFFFF', '#FF00FF', '#00FF00', '#0000FF',
                          '#FF0000', '#00FFFF', '#FF69B4', '#000000', '#FFFFFF', '#FF00FF', '#00FF00', '#0000FF'],
                'description': 'Cool and bright colors - suits clear, bright complexions with cool undertones',
                'characteristics': ['Clear, bright complexion', 'Cool undertones', 'High contrast features'],
                'style': 'Clear and bright with cool undertones'
            }
        }

        # Enhanced skin tone ranges for more accurate detection
        self.skin_tone_ranges = {
            'depth': {
                'Deep': (0, 25),
                'Medium-Deep': (25, 45),
                'Medium': (45, 65),
                'Medium-Light': (65, 85),
                'Light': (85, 100)
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

    def get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Get dominant colors from image using color frequency analysis"""
        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image data
        image_data = np.array(image)
        pixels = image_data.reshape(-1, 3)
        
        # Count color frequencies
        color_counts = Counter(map(tuple, pixels))
        
        # Get most common colors
        dominant_colors = [color for color, _ in color_counts.most_common(num_colors)]
        
        return dominant_colors

    def analyze_skin_tone(self, color: Tuple[int, int, int]) -> Dict:
        """Analyze skin tone characteristics"""
        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        h = h * 360  # Convert hue to degrees
        
        # Determine depth
        depth = v * 100
        depth_category = next((k for k, (min_val, max_val) in self.skin_tone_ranges['depth'].items()
                             if min_val <= depth <= max_val), 'Medium')
        
        # Determine warmth
        warmth_category = next((k for k, ranges in self.skin_tone_ranges['warmth'].items()
                              if any(min_val <= h <= max_val for min_val, max_val in ranges)), 'Neutral')
        
        # Determine clarity
        clarity = s * 100
        clarity_category = next((k for k, (min_val, max_val) in self.skin_tone_ranges['clarity'].items()
                               if min_val <= clarity <= max_val), 'Soft')
        
        return {
            'depth': depth_category,
            'warmth': warmth_category,
            'clarity': clarity_category,
            'hsv': (h, s * 100, v * 100)
        }

    def determine_season(self, skin_analysis: Dict) -> str:
        """Determine season based on skin analysis"""
        depth = skin_analysis['depth']
        warmth = skin_analysis['warmth']
        clarity = skin_analysis['clarity']
        
        # Spring types
        if warmth in ['Warm', 'Neutral-Warm'] and clarity in ['Clear', 'Very Clear']:
            if depth in ['Light', 'Medium-Light']:
                return 'Light Spring'
            elif depth in ['Medium', 'Medium-Deep']:
                return 'True Spring'
            else:
                return 'Clear Spring'
        
        # Summer types
        elif warmth in ['Cool', 'Neutral-Cool'] and clarity in ['Muted', 'Soft']:
            if depth in ['Light', 'Medium-Light']:
                return 'Light Summer'
            elif depth in ['Medium', 'Medium-Deep']:
                return 'True Summer'
            else:
                return 'Soft Summer'
        
        # Autumn types
        elif warmth in ['Warm', 'Neutral-Warm'] and clarity in ['Muted', 'Soft']:
            if depth in ['Light', 'Medium-Light']:
                return 'Soft Autumn'
            elif depth in ['Medium', 'Medium-Deep']:
                return 'True Autumn'
            else:
                return 'Deep Autumn'
        
        # Winter types
        else:
            if clarity in ['Clear', 'Very Clear']:
                return 'Clear Winter'
            elif depth in ['Deep', 'Medium-Deep']:
                return 'Deep Winter'
            else:
                return 'Cool Winter'

    def get_personalized_palette(self, season: str) -> Dict:
        """Get personalized color palette based on season"""
        return self.seasonal_palettes.get(season, self.seasonal_palettes['True Spring'])

    def create_color_palette_visualization(self, colors: List[str], title: str) -> plt.Figure:
        """Create a beautiful color palette visualization"""
        fig, ax = plt.subplots(figsize=(12, 2))
        for i, color in enumerate(colors):
            ax.add_patch(Rectangle((i/len(colors), 0), 1/len(colors), 1, color=color))
        ax.set_title(title, fontsize=14, pad=20)
        ax.axis('off')
        return fig

    def visualize_results(self, image_path: str, results: Dict) -> None:
        """Enhanced visualization with beautiful design"""
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
        colors = results['palette']['colors']
        for i, color in enumerate(colors):
            ax2.add_patch(Rectangle((i/len(colors), 0), 1/len(colors), 1, color=color))
        ax2.set_title('Recommended Colors', fontsize=14, pad=20)
        ax2.axis('off')
        
        # Skin tone analysis
        ax3 = fig.add_subplot(gs[1, :])
        analysis = results['skin_analysis']
        characteristics = results['palette']['characteristics']
        style = results['palette']['style']
        
        info_text = f"""
        Season: {results['season']}
        
        Skin Tone Analysis:
        - Depth: {analysis['depth']}
        - Warmth: {analysis['warmth']}
        - Clarity: {analysis['clarity']}
        
        Characteristics:
        {chr(10).join(f'- {char}' for char in characteristics)}
        
        Style Guide:
        {style}
        
        Color Palette Description:
        {results['palette']['description']}
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
            
            # Determine season with enhanced determination
            season = self.determine_season(skin_analysis)
            
            # Get personalized palette with enhanced recommendations
            palette = self.get_personalized_palette(season)
            
            return {
                'skin_tone': dominant_color,
                'skin_analysis': skin_analysis,
                'season': season,
                'palette': palette
            }
        except Exception as e:
            return {'error': str(e)}

def main():
    analyzer = EnhancedColorAnalyzer()
    
    print("Enhanced Color Analysis System (12-Tone Seasonal Color System)")
    print("============================================================")
    image_path = input("\nEnter the path to your image: ")
    
    print("\nAnalyzing image...")
    results = analyzer.analyze_image(image_path)
    
    if 'error' not in results:
        print("\nAnalysis Results:")
        print("----------------")
        print(f"Season: {results['season']}")
        print("\nSkin Tone Analysis:")
        print(f"- Depth: {results['skin_analysis']['depth']}")
        print(f"- Warmth: {results['skin_analysis']['warmth']}")
        print(f"- Clarity: {results['skin_analysis']['clarity']}")
        
        print("\nCharacteristics:")
        for char in results['palette']['characteristics']:
            print(f"- {char}")
        
        print("\nStyle Guide:")
        print(results['palette']['style'])
        
        print("\nRecommended Colors:")
        for color in results['palette']['colors']:
            print(f"- {color}")
        
        print("\nColor Palette Description:")
        print(results['palette']['description'])
        
        # Visualize results
        analyzer.visualize_results(image_path, results)
    else:
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()