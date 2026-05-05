"""
Visualization helpers using plain text only.
No external libraries.
"""

from dataset import CLASS_NAMES


def print_title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_subtitle(text):
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def matrix_to_text(matrix, precision=2, limit_rows=None, limit_cols=None):
    rows = matrix
    if limit_rows is not None:
        rows = rows[:limit_rows]

    lines = []
    for row in rows:
        values = row
        if limit_cols is not None:
            values = values[:limit_cols]
        lines.append(" ".join(f"{value:6.{precision}f}" for value in values))

    return "\n".join(lines)


def print_matrix(matrix, name="Matrix", precision=2, limit_rows=None, limit_cols=None):
    print_subtitle(name)
    print(matrix_to_text(matrix, precision=precision, limit_rows=limit_rows, limit_cols=limit_cols))


def ascii_pixel(value):
    if value >= 0.85:
        return "#"
    if value >= 0.45:
        return "+"
    if value >= 0.15:
        return "-"
    return "."


def print_ascii_image(image_2d, name="ASCII image"):
    print_subtitle(name)
    for row in image_2d:
        print(" ".join(ascii_pixel(value) for value in row))


def print_digit_image(image_2d, name="Image as 0/1 digits"):
    print_subtitle(name)
    for row in image_2d:
        print(" ".join("1" if value > 0.5 else "0" for value in row))


def print_filters(conv_layer, name="Conv kernels / filters"):
    print_title(name)
    for f, filt in enumerate(conv_layer.weights):
        print(f"\nFilter {f}")
        for c, channel_kernel in enumerate(filt):
            print(f"Channel {c}")
            print(matrix_to_text(channel_kernel, precision=2))


def print_feature_maps(feature_maps, name="Feature maps", max_maps=4):
    """
    feature_maps shape: [batch][channel][height][width]
    Prints feature maps for the first image in the batch.
    """
    print_title(name)
    first = feature_maps[0]

    for channel_index, fmap in enumerate(first[:max_maps]):
        print_matrix(fmap, name=f"Feature map {channel_index} - numbers", precision=2)
        print_ascii_image(fmap, name=f"Feature map {channel_index} - ASCII")


def print_relu_comparison(before_relu, after_relu, channel=0):
    print_title("ReLU comparison: before vs after")
    before = before_relu[0][channel]
    after = after_relu[0][channel]
    print_matrix(before, name=f"Before ReLU - channel {channel}", precision=2)
    print_matrix(after, name=f"After ReLU - channel {channel}", precision=2)


def print_maxpool_steps(pool_layer, max_steps=8, batch=0, channel=0):
    print_title("MaxPool step-by-step")

    x = pool_layer.input
    out = pool_layer.output
    pool = pool_layer.pool_size
    stride = pool_layer.stride

    _, _, out_h, out_w = len(out), len(out[0]), len(out[0][0]), len(out[0][0][0])

    shown = 0
    for i in range(out_h):
        for j in range(out_w):
            if shown >= max_steps:
                print(f"\n... showing first {max_steps} pooling windows only")
                return

            row = i * stride
            col = j * stride

            window = []
            values = []
            for pi in range(pool):
                row_values = []
                for pj in range(pool):
                    value = x[batch][channel][row + pi][col + pj]
                    row_values.append(value)
                    values.append(value)
                window.append(row_values)

            print(f"\nWindow at output({i},{j})")
            print(matrix_to_text(window, precision=2))
            print(f"Max = {max(values):.2f}")
            shown += 1


def loss_bar(loss, max_loss=2.0, width=30):
    filled = int(max(0.0, min(loss / max_loss, 1.0)) * width)
    return "█" * filled + "░" * (width - filled)


def print_loss_history(loss_history):
    print_title("Loss history")
    max_loss = max(loss_history) if loss_history else 1.0

    for epoch, loss in enumerate(loss_history, start=1):
        print(f"Epoch {epoch:02d}: {loss_bar(loss, max_loss=max_loss)} {loss:.4f}")


def print_probabilities(probabilities):
    print_title("Softmax probabilities")
    probs = probabilities[0]

    for name, value in zip(CLASS_NAMES, probs):
        percent = value * 100
        print(f"{name:12s}: {percent:6.2f}%")


def confusion_matrix(true_labels, predicted_labels, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for true, pred in zip(true_labels, predicted_labels):
        matrix[true][pred] += 1

    return matrix


def print_confusion_matrix(matrix):
    print_title("Confusion Matrix")

    header = "True \\ Pred".ljust(14)
    for name in CLASS_NAMES:
        header += name[:6].rjust(8)
    print(header)

    for i, row in enumerate(matrix):
        line = CLASS_NAMES[i][:12].ljust(14)
        for value in row:
            line += str(value).rjust(8)
        print(line)


def print_model_pipeline():
    print_title("CNN pipeline")
    print("Input image 8x8")
    print("  -> Conv2D: detects small patterns")
    print("  -> ReLU: removes negative values")
    print("  -> MaxPool: keeps strongest features and reduces size")
    print("  -> Flatten: converts feature maps to one list")
    print("  -> Dense: makes final class decision")
    print("  -> Softmax: converts scores to probabilities")
