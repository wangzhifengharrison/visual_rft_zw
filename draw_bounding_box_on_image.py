from PIL import Image, ImageDraw

# === Load image ===
image_path = "/scratch/kf09/zw4360/Visual-RFT/share_data/Inference_data/coco/val2017/000000275791.jpg"  # change to your image path
image = Image.open(image_path).convert("RGB")

# === Define bounding box ===
bbox = [357.76000000000005, 261.324, 117.75999999999993, 100.77199999999999]

# === Fix coordinate order ===
x0, y0, x1, y1 = bbox
x0, x1 = min(x0, x1), max(x0, x1)
y0, y1 = min(y0, y1), max(y0, y1)
fixed_bbox = [x0, y0, x1, y1]

# === Draw the box ===
draw = ImageDraw.Draw(image)
draw.rectangle(fixed_bbox, outline="red", width=3)

# === Optional: draw label ===
label = "cat"
draw.text((x0, y0 - 10), label, fill="red")

# === Show or save the result ===
image.save("000000275791.jpg")