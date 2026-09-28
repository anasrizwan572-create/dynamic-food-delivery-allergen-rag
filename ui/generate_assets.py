"""Generate local SVG assets for each food category."""
from pathlib import Path

ASSET_DIR = Path("ui/assets/food")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "chicken": {
        "color1": "#FF6B6B",
        "color2": "#EE5253",
        "icon": "🍗",
        "title": "Poultry & Chicken",
        "badge": "Freshly Grilled & Roasted"
    },
    "beef": {
        "color1": "#842A1C",
        "color2": "#54160E",
        "icon": "🥩",
        "title": "Prime Beef & Steaks",
        "badge": "Slow Cooked & Tender"
    },
    "bbq": {
        "color1": "#D9480F",
        "color2": "#A61E4D",
        "icon": "🍢",
        "title": "BBQ & Tikka Grills",
        "badge": "Charcoal Smoked"
    },
    "vegetarian": {
        "color1": "#2B8A3E",
        "color2": "#2F9E44",
        "icon": "🥗",
        "title": "Vegetarian & Daal",
        "badge": "Fresh Organic Herbs"
    },
    "seafood": {
        "color1": "#1098AD",
        "color2": "#0C8599",
        "icon": "🍤",
        "title": "Fresh Seafood",
        "badge": "Catch of the Day"
    },
    "fast_food": {
        "color1": "#F59F00",
        "color2": "#E67700",
        "icon": "🍔",
        "title": "Burgers & Pizza",
        "badge": "Crisp & Oven Baked"
    },
    "dessert": {
        "color1": "#D6336C",
        "color2": "#A61E4D",
        "icon": "🍨",
        "title": "Artisanal Desserts",
        "badge": "Traditional Sweets"
    },
    "rice": {
        "color1": "#F76707",
        "color2": "#D9480F",
        "icon": "🍚",
        "title": "Biryani & Pulao",
        "badge": "Fragrant Basmati Rice"
    },
    "bread": {
        "color1": "#E8590C",
        "color2": "#D9480F",
        "icon": "🫓",
        "title": "Tandoori Breads",
        "badge": "Clay Oven Baked"
    },
    "soup": {
        "color1": "#495057",
        "color2": "#343A40",
        "icon": "🍲",
        "title": "Soups & Broths",
        "badge": "Warm & Nourishing"
    },
    "beverage": {
        "color1": "#20C997",
        "color2": "#0CA678",
        "icon": "🥤",
        "title": "Craft Beverages",
        "badge": "Chilled & Refreshing"
    },
    "default": {
        "color1": "#4C6EF5",
        "color2": "#364FC7",
        "icon": "🍽️",
        "title": "Chef's Special",
        "badge": "Gourmet Selection"
    }
}

TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" width="100%" height="100%">
  <defs>
    <linearGradient id="grad_{name}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{color1};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{color2};stop-opacity:1" />
    </linearGradient>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000000" flood-opacity="0.25"/>
    </filter>
  </defs>
  <rect width="600" height="400" rx="16" fill="url(#grad_{name})" />
  <circle cx="300" cy="170" r="95" fill="rgba(255, 255, 255, 0.15)" />
  <circle cx="300" cy="170" r="75" fill="rgba(255, 255, 255, 0.22)" filter="url(#shadow)" />
  <text x="300" y="200" font-size="76" text-anchor="middle" font-family="'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif">{icon}</text>
  <rect x="180" y="275" width="240" height="32" rx="16" fill="rgba(0, 0, 0, 0.25)" />
  <text x="300" y="296" font-size="14" font-weight="600" fill="#FFFFFF" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" letter-spacing="1">{badge}</text>
  <text x="300" y="345" font-size="24" font-weight="700" fill="#FFFFFF" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif">{title}</text>
</svg>
"""

for name, cfg in CATEGORIES.items():
    svg_content = TEMPLATE.format(
        name=name,
        color1=cfg["color1"],
        color2=cfg["color2"],
        icon=cfg["icon"],
        title=cfg["title"],
        badge=cfg["badge"]
    )
    svg_path = ASSET_DIR / f"{name}.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Generated: {svg_path}")

print("All food SVG assets generated successfully.")
