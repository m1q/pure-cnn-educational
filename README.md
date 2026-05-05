# Pure Python Educational CNN

This project implements a small Convolutional Neural Network from scratch using **pure Python only**.

No NumPy. No PyTorch. No TensorFlow.

It is designed for learning what happens inside CNNs.

## What it learns

The model learns to classify simple 8x8 images:

| Class | Meaning |
|---|---|
| 0 | vertical line |
| 1 | horizontal line |
| 2 | diagonal line |
| 3 | X shape |
| 4 | box |
| 5 | plus shape |

## CNN pipeline

```text
Input image 8x8
  -> Conv2D
  -> ReLU
  -> MaxPool
  -> Flatten
  -> Dense
  -> Softmax
```

## Files

```text
pure_cnn_educational/
├── main.py        # Training and demo runner
├── layers.py      # CNN layers from scratch
├── dataset.py     # 8x8 shape dataset generator
├── visualize.py   # ASCII/text visualization tools
└── README.md      # This explanation
```

## Run

```bash
python main.py
```

## Run with full internal explanation

```bash
python main.py --debug
```

This prints:

- Input image as matrix
- Input image as ASCII
- Conv feature maps
- Before/after ReLU
- MaxPool step-by-step
- Flatten output preview
- Dense logits
- Softmax probabilities

## Manual input mode

```bash
python main.py --manual
```

Example vertical line:

```text
00010000
00010000
00010000
00010000
00010000
00010000
00010000
00010000
```

## Educational features included

- Print image as matrix
- Print image as ASCII
- Show kernels before training
- Show kernels after training
- Show feature maps
- Compare before and after ReLU
- Show MaxPool step by step
- Debug mode
- More classes: vertical, horizontal, diagonal, X, box, plus
- Extra noise in the dataset
- Loss history bar chart using text
- Softmax probabilities
- Confusion matrix
- Manual prediction mode
- Bilingual-style comments and simple explanations inside the code

## Notes

This project is intentionally slow compared with NumPy or PyTorch because it uses nested Python lists and loops.

That is the point: it helps you see the CNN under the hood.
