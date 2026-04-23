import os
import sys

# Attempt to import linedraw. Ensure linedraw.py is in the same directory!
try:
    _linedraw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test")
    if _linedraw_dir not in sys.path:
        sys.path.insert(0, _linedraw_dir)
    import linedraw  # pyright: ignore[reportMissingImports]
except ImportError:
    print("Error: Could not import linedraw. Make sure linedraw.py is in this directory.")
    sys.exit(1)

# --- Configuration Constants ---
DRAW_BOX_CM = 10.0
DRAW_CENTER_X = 0.0
DRAW_CENTER_Y = 15.0

# --- Helper Functions ---

def _resample_polyline(poly, num_points):
    """
    Basic index-based resampling to reduce point density on very long strokes.
    """
    if len(poly) <= num_points:
        return poly
    step = len(poly) / num_points
    return [poly[int(i * step)] for i in range(num_points)]

def goto_xy(a0, a1, x, y):
    """
    MOCK function for testing. In your real code, this handles inverse kinematics 
    and servo movement. Here, we just return dummy angles to simulate success.
    """
    # Simulate calculating some new angles
    new_a0 = a0 + 0.1
    new_a1 = a1 + 0.1
    return new_a0, new_a1

# --- Core Pipeline Functions ---

def image_to_strokes(image_path, box_cm=DRAW_BOX_CM, center=(DRAW_CENTER_X, DRAW_CENTER_Y), points_per_stroke=80, write_svg=True):
    """
    Convert an image to a list of strokes, scaled to physical cm coordinates.
    """
    print(f"\nProcessing '{image_path}'...")
    
    # 1. Get the polylines from linedraw
    polylines = linedraw.vectorise(
        image_path,
        resolution=512,
        draw_contours=2,
        repeat_contours=1,
        draw_hatch=False,
        repeat_hatch=0,
    )
    
    polylines = [p for p in polylines if len(p) >= 2]
    if not polylines:
        raise ValueError(f"No contours found in {image_path}.")
    print(f"  Found {len(polylines)} strokes.")

    # 2. Scale all strokes from pixel space -> cm
    all_pts = [pt for stroke in polylines for pt in stroke]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    
    scale = box_cm / max(maxx - minx, maxy - miny, 1e-6)
    cx_img = (minx + maxx) / 2.0
    cy_img = (miny + maxy) / 2.0
    cx, cy = center

    def px_to_cm(px, py):
        return (
            cx + (px - cx_img) * scale,
            cy - (py - cy_img) * scale,   # flip y: image y-down -> robot y-up
        )

    # 3. Convert and resample
    strokes_cm = []
    for poly in polylines:
        poly_cm = [px_to_cm(p[0], p[1]) for p in poly]
        if len(poly_cm) > points_per_stroke:
            poly_cm = _resample_polyline(poly_cm, points_per_stroke)
        strokes_cm.append(poly_cm)
        
    print(f"  Total waypoints: {sum(len(s) for s in strokes_cm)}.")

    if write_svg:
        svg_path = os.path.splitext(image_path)[0] + ".strokes.svg"
        _write_strokes_svg(strokes_cm, svg_path)

    return strokes_cm

def _write_strokes_svg(strokes, svg_path, padding_cm=1.0):
    """
    Write an SVG showing each stroke as a coloured polyline.
    """
    if not strokes:
        return

    all_pts = [pt for s in strokes for pt in s]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    
    minx = min(xs) - padding_cm
    maxx = max(xs) + padding_cm
    miny = min(ys) - padding_cm
    maxy = max(ys) + padding_cm
    
    w, h = maxx - minx, maxy - miny

    colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]
    
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}cm" height="{h:.2f}cm" viewBox="{minx:.3f} {-maxy:.3f} {w:.3f} {h:.3f}">',
        f'<rect x="{minx:.3f}" y="{-maxy:.3f}" width="{w:.3f}" height="{h:.3f}" fill="white" stroke="#ccc" stroke-width="0.02"/>',
    ]

    for i, stroke in enumerate(strokes):
        color = colors[i % len(colors)]
        pts = " ".join(f"{x:.3f},{-y:.3f}" for x, y in stroke)
        parts.append(f'<polyline points="{pts}" stroke="{color}" stroke-width="0.04" fill="none" opacity="0.8"/>')
        x0, y0 = stroke[0]
        parts.append(f'<circle cx="{x0:.3f}" cy="{-y0:.3f}" r="0.08" fill="{color}"/>')

    parts.append("</svg>")

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"  Wrote SVG preview to: {svg_path}")

def trace_strokes(a0, a1, strokes, pen_up_fn=None, pen_down_fn=None):
    """
    Simulate tracing the strokes.
    """
    if not strokes:
        return a0, a1

    print(f"  Simulating trace of {len(strokes)} strokes...")
    for si, stroke in enumerate(strokes):
        if pen_up_fn: pen_up_fn()
        
        x0, y0 = stroke[0]
        a0, a1 = goto_xy(a0, a1, x0, y0)
        
        if pen_down_fn: pen_down_fn()
        
        for x, y in stroke[1:]:
            a0, a1 = goto_xy(a0, a1, x, y)
            
    if pen_up_fn: pen_up_fn()
    return a0, a1

# --- Main Interactive Loop ---

if __name__ == "__main__":
    print("=== Image to Strokes Test Environment ===")
    print("Type 'q' or 'quit' to exit.")
    
    # Mock starting angles for the simulation
    angle0, angle1 = 90.0, 90.0 
    
    # Mock pen lift functions just to show they trigger
    def mock_pen_up(): pass
    def mock_pen_down(): pass

    while True:
        try:
            image_path = input("\nEnter path to image: ").strip()
        except EOFError:
            break
            
        if image_path.lower() in ['q', 'quit', 'exit']:
            break
            
        if not image_path:
            continue
            
        if not os.path.exists(image_path):
            print(f"  Error: File '{image_path}' not found.")
            continue

        try:
            # 1. Convert image to strokes and generate SVG
            strokes = image_to_strokes(image_path)
            
            # 2. Simulate the drawing process
            angle0, angle1 = trace_strokes(angle0, angle1, strokes, pen_up_fn=mock_pen_up, pen_down_fn=mock_pen_down)
            
            print(f"  Trace simulation complete. Final mock angles: a0={angle0:.2f}, a1={angle1:.2f}")
            
        except Exception as e:
            print(f"  Failed to process image: {e}")