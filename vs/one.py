import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import colorsys

class EnhancedColorAnalyzer:
    def __init__(self):
        # Enhanced seasonal color palettes with more specific tones
        self.seasonal_palettes = {
            'Deep Winter': {
                'colors': ['#000000', '#1B1B1B', '#2C2C2C', '#3D3D3D', '#4E4E4E', '#1A237E', '#0D47A1', '#01579B'],
                'description': 'Deep, rich, and cool colors with high contrast'
            },
            'True Winter': {
                'colors': ['#000080', '#0000CD', '#0000FF', '#1E90FF', '#00BFFF', '#87CEEB', '#B0E0E6', '#F0F8FF'],
                'description': 'Clear, cool, and bright colors'
            },
            'Bright Spring': {
                'colors': ['#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500', '#FF8C00', '#FFA500'],
                'description': 'Warm, bright, and clear colors'
            },
            'True Spring': {
                'colors': ['#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500', '#FF8C00', '#FFA500'],
                'description': 'Warm and clear colors with medium contrast'
            },
            'Light Summer': {
                'colors': ['#87CEEB', '#B0E0E6', '#ADD8E6', '#B0C4DE', '#A9A9A9', '#D8BFD8', '#DDA0DD', '#EE82EE'],
                'description': 'Soft, cool, and light colors'
            },
            'True Summer': {
                'colors': ['#4682B4', '#5F9EA0', '#20B2AA', '#48D1CC', '#40E0D0', '#7FFFD4', '#66CDAA', '#3CB371'],
                'description': 'Soft, cool, and muted colors'
            },
            'Soft Autumn': {
                'colors': ['#8B4513', '#A0522D', '#6B8E23', '#556B2F', '#8B6914', '#CD853F', '#DEB887', '#D2B48C'],
                'description': 'Soft, warm, and muted colors'
            },
            'True Autumn': {
                'colors': ['#8B4513', '#A0522D', '#6B8E23', '#556B2F', '#8B6914', '#CD853F', '#DEB887', '#D2B48C'],
                'description': 'Warm and muted colors with medium contrast'
            }
        }

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV with enhanced precision"""
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100

    def analyze_skin_tone(self, rgb):
        """Enhanced skin tone analysis"""
        h, s, v = self.rgb_to_hsv(rgb)
        
        # Determine skin tone depth
        if v < 30:
            depth = "Deep"
        elif v < 60:
            depth = "Medium"
        else:
            depth = "Light"
            
        # Determine skin tone warmth
        if h < 30 or h > 330:
            warmth = "Warm"
        elif h < 60 or h > 300:
            warmth = "Neutral-Warm"
        elif h < 120 or h > 240:
            warmth = "Neutral-Cool"
        else:
            warmth = "Cool"
            
        # Determine skin tone clarity
        if s < 20:
            clarity = "Muted"
        elif s < 40:
            clarity = "Soft"
        else:
            clarity = "Clear"
            
        return {
            'depth': depth,
            'warmth': warmth,
            'clarity': clarity,
            'hsv': (h, s, v)
        }

    def get_dominant_colors(self, image, num_colors=5):
        """Enhanced dominant color detection"""
        # Convert image to RGB if it isn't already
        image = image.convert('RGB')
        
        # Resize image for faster processing
        image = image.resize((100, 100), Image.Resampling.LANCZOS)
        
        # Get all pixels
        pixels = list(image.getdata())
        
        # Filter for skin tones with more precise ranges
        skin_pixels = []
        for pixel in pixels:
            h, s, v = self.rgb_to_hsv(pixel)
            if (20 <= v <= 95 and  # Value range
                10 <= s <= 90 and  # Saturation range
                0 <= h <= 50):     # Hue range
                skin_pixels.append(pixel)
        
        if not skin_pixels:
            raise ValueError("No skin tones detected in the image")
        
        # Count pixel frequencies
        pixel_counts = Counter(skin_pixels)
        
        # Get most common colors
        dominant_colors = pixel_counts.most_common(num_colors)
        
        # Calculate weighted average color
        total_weight = sum(count for _, count in dominant_colors)
        avg_color = np.zeros(3)
        for color, count in dominant_colors:
            weight = count / total_weight
            avg_color += np.array(color) * weight
        
        return avg_color.astype(int)

    def determine_season(self, skin_analysis):
        """Determine seasonal color type based on skin analysis"""
        depth = skin_analysis['depth']
        warmth = skin_analysis['warmth']
        clarity = skin_analysis['clarity']
        
        if depth == "Deep":
            if "Cool" in warmth:
                return "Deep Winter"
            else:
                return "Soft Autumn"
        elif depth == "Light":
            if "Cool" in warmth:
                return "Light Summer"
            else:
                return "Bright Spring"
        else:  # Medium depth
            if "Warm" in warmth:
                if clarity == "Clear":
                    return "True Spring"
                else:
                    return "True Autumn"
            else:
                if clarity == "Clear":
                    return "True Winter"
                else:
                    return "True Summer"

    def get_personalized_palette(self, season):
        """Get personalized color palette based on season"""
        return {
            'season': season,
            'colors': self.seasonal_palettes[season]['colors'],
            'description': self.seasonal_palettes[season]['description']
        }

    def analyze_image(self, image_path):
        """Analyze image and return detailed results"""
        try:
            # Open and process image
            image = Image.open(image_path)
            
            # Get dominant color
            dominant_color = self.get_dominant_colors(image)
            
            # Analyze skin tone
            skin_analysis = self.analyze_skin_tone(dominant_color)
            
            # Determine season
            season = self.determine_season(skin_analysis)
            
            # Get personalized palette
            palette = self.get_personalized_palette(season)
            
            return {
                'skin_tone': dominant_color,
                'skin_analysis': skin_analysis,
                'season': season,
                'palette': palette
            }
        except Exception as e:
            return {'error': str(e)}

    def visualize_results(self, image_path, results):
        """Enhanced visualization of analysis results"""
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
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
        info_text = f"""
        Season: {results['season']}
        Skin Tone Analysis:
        - Depth: {analysis['depth']}
        - Warmth: {analysis['warmth']}
        - Clarity: {analysis['clarity']}
        
        Color Palette Description:
        {results['palette']['description']}
        """
        ax3.text(0.1, 0.5, info_text, fontsize=12, va='center')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    analyzer = EnhancedColorAnalyzer()
    
    print("Enhanced Color Analysis System")
    print("=============================")
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