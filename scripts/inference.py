#!/usr/bin/env python3
"""
Single Image Inference Script

Classify exercises from a single image or directory of images.

Usage:
    python scripts/inference.py image.jpg
    python scripts/inference.py images/ --output results.json
    python scripts/inference.py image.jpg --top-k 3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import ExerciseClassifier, EXERCISE_CLASSES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify exercises from images"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Path to image file or directory of images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=1,
        help="Number of top predictions to show (default: 1)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (default: 0.0)"
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List all supported exercise classes and exit"
    )
    return parser.parse_args()


def classify_image(
    classifier: ExerciseClassifier,
    image_path: Path,
    top_k: int = 1,
    threshold: float = 0.0
) -> dict:
    """
    Classify a single image.
    
    Args:
        classifier: ExerciseClassifier instance
        image_path: Path to image file
        top_k: Number of top predictions
        threshold: Minimum confidence threshold
        
    Returns:
        Dictionary with classification results
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {"error": f"Could not read image: {image_path}"}
    
    predictions = classifier.get_top_k_predictions(frame, k=top_k)
    
    # Filter by threshold
    predictions = [(label, conf) for label, conf in predictions if conf >= threshold]
    
    return {
        "file": str(image_path.name),
        "predictions": [
            {"exercise": label, "confidence": round(conf, 4)}
            for label, conf in predictions
        ]
    }


def main() -> None:
    """Run inference on images."""
    args = parse_args()
    
    # List classes and exit
    if args.list_classes:
        print("Supported Exercise Classes (22 total):")
        print("-" * 40)
        for i, cls in enumerate(EXERCISE_CLASSES, 1):
            print(f"  {i:2d}. {cls}")
        return
    
    # Check input is provided
    if args.input is None:
        print("Error: input is required (unless using --list-classes)")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    # Collect image files
    if input_path.is_file():
        image_files = [input_path]
    else:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in extensions
        ]
        image_files.sort()
    
    if not image_files:
        print(f"Error: No image files found in {input_path}")
        sys.exit(1)
    
    print(f"Processing {len(image_files)} image(s)...")
    print()
    
    # Initialize classifier
    classifier = ExerciseClassifier()
    
    results = []
    for image_path in image_files:
        result = classify_image(
            classifier,
            image_path,
            top_k=args.top_k,
            threshold=args.threshold
        )
        results.append(result)
        
        # Print results
        if "error" in result:
            print(f"{image_path.name}: {result['error']}")
        else:
            print(f"{image_path.name}:")
            for pred in result["predictions"]:
                print(f"  {pred['exercise']}: {pred['confidence']:.1%}")
        print()
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
