"""
Educational Pure Python CNN.

Run:
    python main.py

More detailed internal view:
    python main.py --debug

Manual 8x8 input after training:
    python main.py --manual
"""

import argparse
import random

from dataset import CLASS_NAMES, make_dataset, make_one_sample, parse_manual_image
from layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU, SoftmaxCrossEntropy, softmax
from visualize import (
    confusion_matrix,
    print_ascii_image,
    print_confusion_matrix,
    print_digit_image,
    print_feature_maps,
    print_filters,
    print_loss_history,
    print_maxpool_steps,
    print_model_pipeline,
    print_probabilities,
    print_relu_comparison,
    print_subtitle,
    print_title,
)


class EducationalCNN:
    def __init__(self, num_classes):
        self.conv = Conv2D(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = ReLU()
        self.pool = MaxPool2D(pool_size=2, stride=2)
        self.flatten = Flatten()

        # Image is 8x8.
        # Conv with padding=1 keeps it 8x8.
        # MaxPool 2x2 makes it 4x4.
        # Channels = 6.
        # Flatten size = 6 * 4 * 4 = 96.
        self.dense = Dense(in_features=6 * 4 * 4, out_features=num_classes)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        conv_out = self.conv.forward(x)
        relu_out = self.relu.forward(conv_out)
        pool_out = self.pool.forward(relu_out)
        flat_out = self.flatten.forward(pool_out)
        logits = self.dense.forward(flat_out)
        return logits

    def train_step(self, x, labels, learning_rate):
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, labels)

        grad = self.loss_fn.backward()
        grad = self.dense.backward(grad, learning_rate)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        self.conv.backward(grad, learning_rate)

        return loss

    def predict_proba(self, x):
        logits = self.forward(x)
        return softmax(logits)

    def predict(self, x):
        probs = self.predict_proba(x)
        predictions = []

        for row in probs:
            predictions.append(row.index(max(row)))

        return predictions


def accuracy(model, x, labels):
    predictions = model.predict(x)
    correct = 0

    for pred, true in zip(predictions, labels):
        if pred == true:
            correct += 1

    return correct / len(labels)


def run_debug_demo(model, sample_x, sample_y):
    print_title("Step-by-step debug demo")

    image_2d = sample_x[0][0]
    print(f"True label: {sample_y} = {CLASS_NAMES[sample_y]}")
    print_digit_image(image_2d, name="Input image as 0/1 matrix")
    print_ascii_image(image_2d, name="Input image as ASCII")

    logits = model.forward(sample_x)
    probs = softmax(logits)

    print_feature_maps(model.conv.output, name="After Conv2D: feature maps", max_maps=3)
    print_relu_comparison(model.conv.output, model.relu.output, channel=0)
    print_feature_maps(model.pool.output, name="After MaxPool: smaller feature maps", max_maps=3)
    print_maxpool_steps(model.pool, max_steps=8, batch=0, channel=0)

    print_subtitle("Flatten output preview")
    print(model.flatten.output[0][:30], "...")

    print_subtitle("Dense logits")
    print([round(v, 3) for v in logits[0]])

    print_probabilities(probs)

    pred = probs[0].index(max(probs[0]))
    print(f"Prediction: {pred} = {CLASS_NAMES[pred]}")


def manual_prediction_loop(model):
    print_title("Manual prediction mode")
    print("Enter 8 lines. Example for a vertical line:")
    print("00010000")
    print("00010000")
    print("00010000")
    print("00010000")
    print("00010000")
    print("00010000")
    print("00010000")
    print("00010000")
    print("Type q to quit.\n")

    while True:
        lines = []
        for i in range(8):
            line = input(f"line {i + 1}: ").strip()
            if line.lower() == "q":
                return
            lines.append(line)

        try:
            x = parse_manual_image(lines)
        except ValueError as error:
            print("Input error:", error)
            continue

        probs = model.predict_proba(x)
        pred = probs[0].index(max(probs[0]))

        print_ascii_image(x[0][0], name="Your input")
        print_probabilities(probs)
        print(f"Prediction: {CLASS_NAMES[pred]}\n")


def main():
    parser = argparse.ArgumentParser(description="Educational CNN from scratch using pure Python.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.15, help="Learning rate.")
    parser.add_argument("--train-size", type=int, default=180, help="Number of training samples.")
    parser.add_argument("--test-size", type=int, default=60, help="Number of test samples.")
    parser.add_argument("--debug", action="store_true", help="Show internal CNN steps for one sample.")
    parser.add_argument("--manual", action="store_true", help="Enter your own 8x8 image after training.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)

    print_model_pipeline()

    train_x, train_y = make_dataset(n=args.train_size)
    test_x, test_y = make_dataset(n=args.test_size)

    sample_x, sample_y = make_one_sample(label=0)
    sample_batch = [sample_x]

    print_title("Dataset example")
    print(f"Example label: {sample_y} = {CLASS_NAMES[sample_y]}")
    print_digit_image(sample_x[0], name="Image as 0/1 matrix")
    print_ascii_image(sample_x[0], name="Image as ASCII")

    model = EducationalCNN(num_classes=len(CLASS_NAMES))

    print_filters(model.conv, name="Conv filters BEFORE training")

    print_title("Training")
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        loss = model.train_step(train_x, train_y, learning_rate=args.lr)
        loss_history.append(loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            acc = accuracy(model, test_x, test_y)
            print(f"Epoch {epoch:02d} | loss={loss:.4f} | test accuracy={acc:.2%}")

    print_loss_history(loss_history)

    print_filters(model.conv, name="Conv filters AFTER training")

    test_predictions = model.predict(test_x)
    matrix = confusion_matrix(test_y, test_predictions, len(CLASS_NAMES))
    print_confusion_matrix(matrix)

    probs = model.predict_proba(sample_batch)
    pred = probs[0].index(max(probs[0]))

    print_title("Sample prediction after training")
    print_digit_image(sample_x[0], name="Sample input")
    print_probabilities(probs)
    print(f"Prediction: {pred} = {CLASS_NAMES[pred]}")
    print(f"True label: {sample_y} = {CLASS_NAMES[sample_y]}")

    # Always show a light educational view.
    # --debug shows a deeper and longer one.
    if args.debug:
        run_debug_demo(model, sample_batch, sample_y)
    else:
        print_title("Tip")
        print("Run this for full internal explanation:")
        print("python main.py --debug")
        print("Run this to type your own 8x8 shape:")
        print("python main.py --manual")

    if args.manual:
        manual_prediction_loop(model)


if __name__ == "__main__":
    main()
