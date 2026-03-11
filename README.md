# Hungarian Card Dataset Generator

This project generates a synthetic dataset of Hungarian (Belote) playing cards for training object detection models with the [Darknet / YOLO](https://codeberg.org/CCodeRun/darknet/) framework.

The generator places cropped card images onto random background textures, applies random transformations (scale, rotation, placement), and produces:

- training images
- YOLO-compatible label files (`.txt`)

The resulting dataset can be used to train a detector that identifies individual card types.

---

## Requirements

The generator was developed and tested with:

- **Python 3.13**

Older Python versions will likely work as well.

Required Python packages can be installed with `pip install -r ./requirements.txt`

## Dataset Source Structure

Before running the generator, the dataset_source directory must contain:

```bash
dataset_source/
├── cards/
│   ├── 10h/
│   │   └── 10h.png
│   ├── 10k/
│   │   └── 10k.png
│   └── ...
└── backgrounds/
    ├── braided/
    ├── bumpy/
    ├── chequered/
    └── ...
```

### Cards

Each card must be a cropped PNG with transparency and placed inside a directory named after the class.

### Backgrounds

The backgrounds directory should contain random images used as scene backgrounds.

Subdirectories are supported and recommended for organization.

Example:
```bash
backgrounds/
├── braided/
├── bumpy/
├── chequered/
└── ...
```

In this project the Describable Textures Dataset ([DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/)) was used, but any sufficiently varied set of images should work.

### Output

The generator creates the following structure:
```bash
output/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each generated image has a corresponding YOLO label file containing bounding boxes and class IDs.
