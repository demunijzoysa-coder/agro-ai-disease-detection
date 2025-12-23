import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASS_NAMES = ["blast", "healthy"]

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)

def predict(model, image_path: Path, threshold=0.5):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    prob_healthy = float(model.predict(x, verbose=0)[0][0])
    prob_blast = 1.0 - prob_healthy

    predicted = "healthy" if prob_healthy >= threshold else "blast"

    return {
        "image": str(image_path),
        "predicted": predicted,
        "prob_blast": prob_blast,
        "prob_healthy": prob_healthy,
    }


def main():
    parser = argparse.ArgumentParser(description="Rice Leaf Blast inference (binary: blast vs healthy).")
    parser.add_argument("image", help="Path to an image file (jpg/png).")
    parser.add_argument("--model", default=str(Path("models") / "rice_leaf_blast_cnn.keras"),
                        help="Path to saved .keras model (default: models/rice_leaf_blast_cnn.keras)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for predicting healthy (prob_healthy >= threshold => healthy). Default: 0.5")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    model = load_model(model_path)
    out = predict(model, image_path, threshold=args.threshold)

    print("\n=== Prediction ===")
    print("Image      :", out["image"])
    print("Predicted  :", out["predicted"])
    print("P(blast)   :", f'{out["prob_blast"]:.4f}')
    print("P(healthy) :", f'{out["prob_healthy"]:.4f}')
    print("Threshold  :", args.threshold)


if __name__ == "__main__":
    main()
