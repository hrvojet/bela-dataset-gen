import os
import random
import math
import time
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# =========================
# CONFIG
# =========================
CARDS_DIR = Path("dataset_source/cards")
BACKGROUNDS_DIR = Path("dataset_source/backgrounds")
OUTPUT_DIR = Path("output")

IMAGE_SIZE = (1280, 720)
TRAIN_SPLIT = 0.9
NUM_IMAGES = 50

MIN_CARDS_PER_IMAGE = 1
MAX_CARDS_PER_IMAGE = 4

SCALE_RANGE = (0.18, 0.38)   # relative to background width
ROTATION_RANGE = (-25, 25)
MAX_TRIES_PER_CARD = 30

JPEG_QUALITY = 80

random.seed()

# =========================
# HELPERS
# =========================
def ensure_dirs():
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def list_backgrounds():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in BACKGROUNDS_DIR.rglob("*") if p.suffix.lower() in exts]


def load_cards_to_memory(card_samples):
    cc = []
    for img_path, class_id in card_samples:
        img = Image.open(img_path).convert("RGBA")
        cc.append((img, class_id))

    return cc


def load_bg_to_memory(backgrounds):
    bg_mem = []
    el_no = len(backgrounds)
    q = el_no / 8
    for i, bg_path in enumerate(backgrounds):
        if i % q == 0 and i != 0:
            print(i, "out of ", el_no, " backgrounds loaded")
        bg = Image.open(bg_path).convert("RGBA")
        bg = resize_background(bg)
        bg_mem.append(bg)

    return bg_mem

def build_class_map():
    """
    Assumes each subfolder in CARDS_DIR is one class.
    Example:
      cards/
        7h/
        7k/
    """
    class_dirs = [p for p in CARDS_DIR.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name)

    class_map = {}
    samples = []

    for class_id, class_dir in enumerate(class_dirs):
        class_map[class_dir.name] = class_id
        for img_path in class_dir.rglob("*.png"):
            samples.append((img_path, class_id))

    return class_map, samples

def load_rgba(path):
    return Image.open(path).convert("RGBA")

def resize_background(bg):
    return bg.resize(IMAGE_SIZE, Image.Resampling.LANCZOS).convert("RGB")

def random_background(backgrounds):
    bg_path = random.choice(backgrounds)
    bg = Image.open(bg_path).convert("RGB")
    return resize_background(bg)

def augment_card(card_rgba, bg_w, bg_h):
    # Random scale based on background width
    target_w = int(bg_w * random.uniform(*SCALE_RANGE))
    ratio = target_w / card_rgba.width
    target_h = int(card_rgba.height * ratio)
    card = card_rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Slight random color / brightness
    if random.random() < 0.8:
        enhancer = ImageEnhance.Brightness(card)
        card = enhancer.enhance(random.uniform(0.85, 1.15))

    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(card)
        card = enhancer.enhance(random.uniform(0.9, 1.15))

    # Small blur sometimes
    if random.random() < 0.2:
        card = card.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    # Rotation
    angle = random.uniform(*ROTATION_RANGE)
    card = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    return card

def bbox_from_alpha(card_rgba, x, y):
    """
    Returns visible bounding box from alpha channel after placement.
    """
    alpha = card_rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return None

    left, top, right, bottom = bbox
    return (x + left, y + top, x + right, y + bottom)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    if union == 0:
        return 0
    return inter / union

def yolo_line(class_id, box, img_w, img_h):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def place_card(bg, card_rgba, existing_boxes):
    bg_w, bg_h = bg.size

    for _ in range(MAX_TRIES_PER_CARD):
        x_max = bg_w - card_rgba.width
        y_max = bg_h - card_rgba.height
        if x_max <= 0 or y_max <= 0:
            return None

        x = random.randint(0, x_max)
        y = random.randint(0, y_max)

        box = bbox_from_alpha(card_rgba, x, y)
        if box is None:
            continue

        # limit overlap with previous cards
        too_much_overlap = any(iou(box, prev) > 0.35 for prev in existing_boxes)
        if too_much_overlap:
            continue

        return x, y, box

    return None

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()
    backgrounds = list_backgrounds()
    class_map, card_samples = build_class_map()

    if not backgrounds:
        raise RuntimeError("No background images found.")
    if not card_samples:
        raise RuntimeError("No card PNG files found.")

    cards_cache = load_cards_to_memory(card_samples)
    backgrounds_cache = load_bg_to_memory(backgrounds)

    print("Cached Cards:", len(cards_cache))
    print("Cached backgrounds:", len(backgrounds_cache))
    start = time.time()

    print("Classes:")
    for k, v in class_map.items():
        print(f"  {v}: {k}")

    for i in range(NUM_IMAGES):
        split = "train" if random.random() < TRAIN_SPLIT else "val"

        bg = random.choice(backgrounds_cache).copy()
        bg_rgba = bg.convert("RGBA")

        n_cards = random.randint(MIN_CARDS_PER_IMAGE, MAX_CARDS_PER_IMAGE)
        chosen = random.sample(cards_cache, k=min(n_cards, len(cards_cache)))

        labels = []
        placed_boxes = []

        for card_path, class_id in chosen:
            card = card_path.copy()
            card = augment_card(card, bg_rgba.width, bg_rgba.height)

            placement = place_card(bg_rgba, card, placed_boxes)
            if placement is None:
                continue

            x, y, box = placement
            bg_rgba.alpha_composite(card, (x, y))
            placed_boxes.append(box)
            labels.append(yolo_line(class_id, box, bg_rgba.width, bg_rgba.height))

        # skip empty image
        if not labels:
            continue

        # Optional slight final image effects
        final_img = bg_rgba.convert("RGB")
        if random.random() < 0.15:
            final_img = final_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        stem = f"img_{i:06d}"
        img_path = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
        lbl_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"

        final_img.save(img_path, quality=JPEG_QUALITY, optimize=False)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

    print("\nDone.")
    print(f"Images saved to: {OUTPUT_DIR / 'images'}")
    print(f"Labels saved to: {OUTPUT_DIR / 'labels'}")
    end = time.time()
    print("\nRuntime: ", end - start, " seconds")

if __name__ == "__main__":
    main()