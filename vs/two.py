import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import colorsys
from collections import Counter


class EnhancedColorAnalyzer:
    def __init__(self):
        self.seasonal_palettes = {
            'Light Spring': {
                'colors': ['#FFF5E6', '#FFDAB9', '#B0E0E6', '#FFB6C1', '#F5DEB3'],
                'description': 'Light, warm, and clear colors - perfect for light skin with golden undertones'
            },
            'True Spring': {
                'colors': ['#FFD700', '#FF4500', '#FA8072', '#40E0D0', '#FFA500'],
                'description': 'Medium contrast, warm colors - ideal for medium skin with golden undertones'
            },
            'Clear Spring': {
                'colors': ['#FF69B4', '#32CD32', '#0000CD', '#FF0000', '#FFD700'],
                'description': 'High contrast, warm but bright colors - suits clear, bright complexions'
            },
            'Light Summer': {
                'colors': ['#E6E6FA', '#98FB98', '#B0E0E6', '#FFB6C1', '#F0F8FF'],
                'description': 'Light, cool, and soft colors - perfect for light, ashy skin tones'
            },
            'True Summer': {
                'colors': ['#DDA0DD', '#DEB887', '#000080', '#A9A9A9', '#B0C4DE'],
                'description': 'Cool and muted colors - ideal for cool undertones with medium contrast'
            },
            'Soft Summer': {
                'colors': ['#708090', '#FFB6C1', '#E6E6FA', '#B0C4DE', '#D8BFD8'],
                'description': 'Cool and very muted colors - suits soft, muted complexions'
            },
            'Soft Autumn': {
                'colors': ['#A9A9A9', '#9ACD32', '#DEB887', '#6B8E23', '#D2B48C'],
                'description': 'Warm and muted colors - perfect for warm, muted complexions'
            },
            'True Autumn': {
                'colors': ['#FF7F00', '#D2691E', '#DAA520', '#556B2F', '#8B4513'],
                'description': 'Warm and rich colors - ideal for golden, tan skin tones'
            },
            'Deep Autumn': {
                'colors': ['#800020', '#556B2F', '#8B4513', '#D2691E', '#8B0000'],
                'description': 'Dark, warm, and muted colors - suits deep, rich complexions'
            },
            'Cool Winter': {
                'colors': ['#FF00FF', '#0000CD', '#36454F', '#FFB6C1', '#000000'],
                'description': 'Cool and high contrast colors - perfect for cool undertones with high contrast'
            },
            'Deep Winter': {
                'colors': ['#800080', '#000000', '#DC143C', '#000080', '#4B0082'],
                'description': 'Cool, dark, and clear colors - ideal for deep skin with cool undertones'
            },
            'Clear Winter': {
                'colors': ['#FF0000', '#00FFFF', '#FF69B4', '#000000', '#FFFFFF'],
                'description': 'Cool and bright colors - suits clear, bright complexions with cool undertones'
            }
        }

    def rgb_to_hsv(self, rgb):
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100

    def analyze_skin_tone(self, rgb):
        h, s, v = self.rgb_to_hsv(rgb)
        if v < 25:
            depth = "Deep"
        elif v < 45:
            depth = "Medium-Deep"
        elif v < 65:
            depth = "Medium"
        elif v < 85:
            depth = "Medium-Light"
        else:
            depth = "Light"

        if h < 20 or h > 340:
            warmth = "Warm"
        elif h < 40 or h > 320:
            warmth = "Neutral-Warm"
        elif h < 60 or h > 300:
            warmth = "Neutral"
        elif h < 80 or h > 280:
            warmth = "Neutral-Cool"
        else:
            warmth = "Cool"

        if s < 15:
            clarity = "Very Muted"
        elif s < 30:
            clarity = "Muted"
        elif s < 45:
            clarity = "Soft"
        elif s < 60:
            clarity = "Clear"
        else:
            clarity = "Very Clear"

        return {'depth': depth, 'warmth': warmth, 'clarity': clarity, 'hsv': (h, s, v)}

    def determine_season(self, analysis):
        depth = analysis['depth']
        warmth = analysis['warmth']
        clarity = analysis['clarity']

        if "Warm" in warmth:
            if depth in ["Light", "Medium-Light"]:
                return "Light Spring"
            elif clarity in ["Clear", "Very Clear"]:
                return "Clear Spring"
            else:
                return "True Spring"
        elif "Cool" in warmth:
            if depth in ["Light", "Medium-Light"]:
                return "Light Summer"
            elif clarity in ["Muted", "Very Muted"]:
                return "Soft Summer"
            else:
                return "True Summer"
        elif "Neutral-Warm" in warmth:
            if depth in ["Deep", "Medium-Deep"]:
                return "Deep Autumn"
            elif clarity in ["Muted", "Very Muted"]:
                return "Soft Autumn"
            else:
                return "True Autumn"
        else:
            if depth in ["Deep", "Medium-Deep"]:
                return "Deep Winter"
            elif clarity in ["Clear", "Very Clear"]:
                return "Clear Winter"
            else:
                return "Cool Winter"

    def get_dominant_colors(self, image, resize=100):
        image = image.resize((resize, resize))
        pixels = list(image.getdata())
        pixels = [p for p in pixels if len(p) == 3]
        most_common = Counter(pixels).most_common(1)[0][0]
        return most_common

    def get_personalized_palette(self, season):
        return {
            'season': season,
            'colors': self.seasonal_palettes[season]['colors'],
            'description': self.seasonal_palettes[season]['description']
        }

    def analyze_image(self, image_path):
        try:
            image = Image.open(image_path)
            dominant_color = self.get_dominant_colors(image)
            skin_analysis = self.analyze_skin_tone(dominant_color)
            season = self.determine_season(skin_analysis)
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
        if 'error' in results:
            print(f"Error: {results['error']}")
            return

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        colors = results['palette']['colors']
        for i, color in enumerate(colors):
            ax2.add_patch(plt.Rectangle((i / len(colors), 0), 1 / len(colors), 1, color=color))
        ax2.set_title('Recommended Colors')
        ax2.axis('off')

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
        print("\nRecommended Colors:")
        for color in results['palette']['colors']:
            print(f"- {color}")
        print("\nColor Palette Description:")
        print(results['palette']['description'])

        analyzer.visualize_results(image_path, results)
    else:
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    main()
