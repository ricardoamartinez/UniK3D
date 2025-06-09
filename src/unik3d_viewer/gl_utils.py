import imgui

def hex_to_imvec4(hex_color, alpha=1.0):
    """Converts a hex color string (e.g., "#RRGGBB") to an imgui.Vec4."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    if lv == 6: # RGB
        rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        return imgui.Vec4(rgb[0], rgb[1], rgb[2], alpha)
    elif lv == 8: # RGBA
        rgba = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6))
        return imgui.Vec4(rgba[0], rgba[1], rgba[2], rgba[3])
    else: # Invalid format, return default (e.g., gray)
        print(f"Warning: Invalid hex color format '{hex_color}'. Using gray.")
        return imgui.Vec4(0.5, 0.5, 0.5, 1.0) 