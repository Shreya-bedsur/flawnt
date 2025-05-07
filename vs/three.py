import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import colorsys
from typing import Dict, List, Tuple, Union

class EnhancedColorAnalyzer:
    def __init__(self):
        # Enhanced 12-Tone Seasonal Color System with more specific color ranges
        self.seasonal_palettes = {
            # Spring Types
            'Light Spring': {
                'colors': ['#FFF5E6', '#FFDAB9', '#B0E0E6', '#FFB6C1', '#F5DEB3', '#FFE4B5', '#FAEBD7', '#FFEFD5'],
                'description': 'Light, warm, and clear colors - perfect for light skin with golden undertones',
                'characteristics': ['Light to medium skin with golden undertones', 'Often has freckles', 'Warm, peachy complexion']
            },
            'True Spring': {
                'colors': ['#FFD700', '#FF4500', '#FA8072', '#40E0D0', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347'],
                'description': 'Medium contrast, warm colors - ideal for medium skin with golden undertones',
                'characteristics': ['Medium skin with golden undertones', 'Clear, warm complexion', 'Natural warmth in skin']
            },
            'Clear Spring': {
                'colors': ['#FF69B4', '#32CD32', '#0000CD', '#FF0000', '#FFD700', '#00FF00', '#FF1493', '#00CED1'],
                'description': 'High contrast, warm but bright colors - suits clear, bright complexions',
                'characteristics': ['Clear, bright complexion', 'High contrast features', 'Warm but bright undertones']
            },
            
            # Summer Types
            'Light Summer': {
                'colors': ['#E6E6FA', '#98FB98', '#B0E0E6', '#FFB6C1', '#F0F8FF', '#F5F5F5', '#E0FFFF', '#F0FFFF'],
                'description': 'Light, cool, and soft colors - perfect for light, ashy skin tones',
                'characteristics': ['Light, ashy skin tones', 'Cool undertones', 'Soft, muted features']
            },
            'True Summer': {
                'colors': ['#DDA0DD', '#DEB887', '#000080', '#A9A9A9', '#B0C4DE', '#C0C0C0', '#D8BFD8', '#DDA0DD'],
                'description': 'Cool and muted colors - ideal for cool undertones with medium contrast',
                'characteristics': ['Medium skin with cool undertones', 'Muted features', 'Balanced contrast']
            },
            'Soft Summer': {
                'colors': ['#708090', '#FFB6C1', '#E6E6FA', '#B0C4DE', '#D8BFD8', '#DCDCDC', '#E6E6FA', '#F0F8FF'],
                'description': 'Cool and very muted colors - suits soft, muted complexions',
                'characteristics': ['Soft, muted complexion', 'Cool undertones', 'Low contrast features']
            },
            
            # Autumn Types
            'Soft Autumn': {
                'colors': ['#A9A9A9', '#9ACD32', '#DEB887', '#6B8E23', '#D2B48C', '#BC8F8F', '#CD853F', '#DEB887'],
                'description': 'Warm and muted colors - perfect for warm, muted complexions',
                'characteristics': ['Warm, muted complexion', 'Golden or olive undertones', 'Soft features']
            },
            'True Autumn': {
                'colors': ['#FF7F00', '#D2691E', '#DAA520', '#556B2F', '#8B4513', '#CD853F', '#DEB887', '#D2B48C'],
                'description': 'Warm and rich colors - ideal for golden, tan skin tones',
                'characteristics': ['Golden or tan skin', 'Rich, warm undertones', 'Medium to deep complexion']
            },
            'Deep Autumn': {
                'colors': ['#800020', '#556B2F', '#8B4513', '#D2691E', '#8B0000', '#4B0082', '#800080', '#4B0082'],
                'description': 'Dark, warm, and muted colors - suits deep, rich complexions',
                'characteristics': ['Deep, rich complexion', 'Warm undertones', 'High contrast features']
            },
            
            # Winter Types
            'Cool Winter': {
                'colors': ['#FF00FF', '#0000CD', '#36454F', '#FFB6C1', '#000000', '#4B0082', '#800080', '#000080'],
                'description': 'Cool and high contrast colors - perfect for cool undertones with high contrast',
                'characteristics': ['Cool undertones', 'High contrast features', 'Clear, bright complexion']
            },
            'Deep Winter': {
                'colors': ['#800080', '#000000', '#DC143C', '#000080', '#4B0082', '#191970', '#000080', '#000000'],
                'description': 'Cool, dark, and clear colors - ideal for deep skin with cool undertones',
                'characteristics': ['Deep skin with cool undertones', 'High contrast', 'Clear features']
            },
            'Clear Winter': {
                'colors': ['#FF0000', '#00FFFF', '#FF69B4', '#000000', '#FFFFFF', '#FF00FF', '#00FF00', '#0000FF'],
                'description': 'Cool and bright colors - suits clear, bright complexions with cool undertones',
                'characteristics': ['Clear, bright complexion', 'Cool undertones', 'High contrast features']
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

    def rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV with enhanced precision"""
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100

    def analyze_skin_tone(self, rgb: Tuple[int, int, int]) -> Dict[str, Union[str, Tuple[float, float, float]]]:
        """Enhanced skin tone analysis with more precise ranges"""
        h, s, v = self.rgb_to_hsv(rgb)
        
        # Determine skin tone depth with more precise ranges
        depth = next((k for k, (min_val, max_val) in self.skin_tone_ranges['depth'].items() 
                     if min_val <= v <= max_val), "Medium")
        
        # Determine skin tone warmth with more precise ranges
        warmth = next((k for k, ranges in self.skin_tone_ranges['warmth'].items() 
                      if any(min_val <= h <= max_val for min_val, max_val in ranges)), "Neutral")
        
        # Determine skin tone clarity with more precise ranges
        clarity = next((k for k, (min_val, max_val) in self.skin_tone_ranges['clarity'].items() 
                       if min_val <= s <= max_val), "Soft")
        
        return {
            'depth': depth,
            'warmth': warmth,
            'clarity': clarity,
            'hsv': (h, s, v)
        }

    def get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> np.ndarray:
        """Enhanced dominant color detection with improved skin tone filtering"""
        # Convert image to RGB if it isn't already
        image = image.convert('RGB')
        
        # Resize image for faster processing while maintaining quality
        image = image.resize((200, 200), Image.Resampling.LANCZOS)
        
        # Get all pixels
        pixels = list(image.getdata())
        
        # Enhanced skin tone filtering with more precise ranges
        skin_pixels = []
        for pixel in pixels:
            h, s, v = self.rgb_to_hsv(pixel)
            if (15 <= v <= 95 and  # Value range
                10 <= s <= 90 and  # Saturation range
                0 <= h <= 50):     # Hue range
                skin_pixels.append(pixel)
        
        if not skin_pixels:
            raise ValueError("No skin tones detected in the image")
        
        # Count pixel frequencies with improved weighting
        pixel_counts = Counter(skin_pixels)
        
        # Get most common colors with enhanced weighting
        dominant_colors = pixel_counts.most_common(num_colors)
        
        # Calculate weighted average color with improved algorithm
        total_weight = sum(count for _, count in dominant_colors)
        avg_color = np.zeros(3)
        for color, count in dominant_colors:
            weight = (count / total_weight) ** 2  # Square the weight for more emphasis on dominant colors
            avg_color += np.array(color) * weight
        
        return avg_color.astype(int)

    def determine_season(self, skin_analysis: Dict[str, Union[str, Tuple[float, float, float]]]) -> str:
        """Enhanced seasonal determination with more nuanced analysis"""
        depth = skin_analysis['depth']
        warmth = skin_analysis['warmth']
        clarity = skin_analysis['clarity']
        
        # Spring Types
        if "Warm" in warmth:
            if depth in ["Light", "Medium-Light"]:
                return "Light Spring"
            elif clarity in ["Clear", "Very Clear"]:
                return "Clear Spring"
            else:
                return "True Spring"
        
        # Summer Types
        elif "Cool" in warmth:
            if depth in ["Light", "Medium-Light"]:
                return "Light Summer"
            elif clarity in ["Muted", "Very Muted"]:
                return "Soft Summer"
            else:
                return "True Summer"
        
        # Autumn Types
        elif "Warm" in warmth or "Neutral-Warm" in warmth:
            if depth in ["Deep", "Medium-Deep"]:
                return "Deep Autumn"
            elif clarity in ["Muted", "Very Muted"]:
                return "Soft Autumn"
            else:
                return "True Autumn"
        
        # Winter Types
        else:  # Cool or Neutral-Cool
            if depth in ["Deep", "Medium-Deep"]:
                return "Deep Winter"
            elif clarity in ["Clear", "Very Clear"]:
                return "Clear Winter"
            else:
                return "Cool Winter"

    def get_personalized_palette(self, season: str) -> Dict[str, Union[str, List[str]]]:
        """Get personalized color palette with enhanced recommendations"""
        palette = self.seasonal_palettes[season]
        return {
            'season': season,
            'colors': palette['colors'],
            'description': palette['description'],
            'characteristics': palette['characteristics']
        }

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze image with enhanced error handling and validation"""
        try:
            # Open and validate image
            image = Image.open(image_path)
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
            
            # Get dominant color with enhanced detection
            dominant_color = self.get_dominant_colors(image)
            
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

    def visualize_results(self, image_path: str, results: Dict) -> None:
        """Enhanced visualization with more detailed analysis display"""
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Color palette
        ax2 = fig.add_subplot(gs[0, 1])
        colors = results['palette']['colors']
        for i, color in enumerate(colors):
            ax2.add_patch(plt.Rectangle((i/len(colors), 0), 1/len(colors), 1, color=color))
        ax2.set_title('Recommended Colors')
        ax2.axis('off')
        
        # Skin tone analysis
        ax3 = fig.add_subplot(gs[1, :])
        analysis = results['skin_analysis']
        characteristics = results['palette']['characteristics']
        info_text = f"""
        Season: {results['season']}
        
        Skin Tone Analysis:
        - Depth: {analysis['depth']}
        - Warmth: {analysis['warmth']}
        - Clarity: {analysis['clarity']}
        
        Characteristics:
        {chr(10).join(f'- {char}' for char in characteristics)}
        
        Color Palette Description:
        {results['palette']['description']}
        
        Recommended Colors:
        {', '.join(results['palette']['colors'])}
        """
        ax3.text(0.1, 0.5, info_text, fontsize=12, va='center')
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
        ax4.text(0.1, 0.5, hsv_text, fontsize=12, va='center')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()

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