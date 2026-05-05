"""
Toy dataset generator for educational CNN.
No image files needed.
The image is an 8x8 matrix made from numbers.
"""

import random


CLASS_NAMES = [
    "vertical",
    "horizontal",
    "diagonal",
    "x_shape",
    "box",
    "plus",
]


def empty_image(size, noise=0.05):
    return [
        [random.random() * noise for _ in range(size)]
        for _ in range(size)
    ]


def draw_vertical(img):
    size = len(img)
    col = random.randint(2, size - 3)
    for r in range(size):
        img[r][col] = 1.0


def draw_horizontal(img):
    size = len(img)
    row = random.randint(2, size - 3)
    for c in range(size):
        img[row][c] = 1.0


def draw_diagonal(img):
    size = len(img)
    for i in range(size):
        img[i][i] = 1.0


def draw_x_shape(img):
    size = len(img)
    for i in range(size):
        img[i][i] = 1.0
        img[i][size - 1 - i] = 1.0


def draw_box(img):
    size = len(img)
    top = 2
    bottom = size - 3
    left = 2
    right = size - 3

    for c in range(left, right + 1):
        img[top][c] = 1.0
        img[bottom][c] = 1.0

    for r in range(top, bottom + 1):
        img[r][left] = 1.0
        img[r][right] = 1.0


def draw_plus(img):
    size = len(img)
    center = size // 2

    # small random shift makes the dataset less memorized
    center += random.choice([-1, 0, 1])
    center = max(2, min(size - 3, center))

    for i in range(size):
        img[center][i] = 1.0
        img[i][center] = 1.0


DRAW_FUNCTIONS = [
    draw_vertical,
    draw_horizontal,
    draw_diagonal,
    draw_x_shape,
    draw_box,
    draw_plus,
]


def add_extra_noise(img, probability=0.03, value=0.7):
    """
    Add random bright pixels.
    This forces the CNN to learn the real shape, not perfect clean drawings.
    """
    size = len(img)
    for r in range(size):
        for c in range(size):
            if random.random() < probability and img[r][c] < 1.0:
                img[r][c] = random.random() * value


def make_one_sample(label=None, size=8, noise=0.05, extra_noise=True):
    if label is None:
        label = random.randint(0, len(CLASS_NAMES) - 1)

    img = empty_image(size, noise=noise)
    DRAW_FUNCTIONS[label](img)

    if extra_noise:
        add_extra_noise(img)

    # CNN expects x[channels][height][width]
    return [img], label


def make_dataset(n=600, size=8, noise=0.05, extra_noise=True):
    images = []
    labels = []

    for _ in range(n):
        image, label = make_one_sample(size=size, noise=noise, extra_noise=extra_noise)
        images.append(image)
        labels.append(label)

    return images, labels


def parse_manual_image(lines):
    """
    Convert 8 text lines like 00010000 into CNN input.
    Any non-zero character is treated as 1.
    """
    clean = []

    for line in lines:
        line = line.strip().replace(" ", "")
        if not line:
            continue
        clean.append(line)

    if len(clean) != 8:
        raise ValueError("You must enter exactly 8 non-empty lines.")

    image = []
    for line in clean:
        if len(line) != 8:
            raise ValueError("Each line must contain exactly 8 digits/characters.")

        row = []
        for ch in line:
            if ch in ["0", ".", "_"]:
                row.append(0.0)
            else:
                row.append(1.0)
        image.append(row)

    return [[image]]
