"""
EncodeGenerator.py
===============================================================================
Smart Attendance System - Face Encoding Generator
===============================================================================
Loops through every image inside the `dataset/` folder, extracts the Roll
Number and Name from the filename, computes a 128-dimension face encoding
using the `face_recognition` library, and stores everything in a pickle
file (`Encodings.p`) that the main GUI application (`main.py`) loads at
startup to perform real-time face matching.

Expected image naming convention:
    "RollNo_Name.jpg"   e.g. "101_Rohan.jpg"  ->  Roll No = "101", Name = "Rohan"

Supported image extensions: .jpg, .jpeg, .png

Usage:
    python EncodeGenerator.py

Run this script every time you add/remove images from the `dataset/`
folder so that `Encodings.p` stays in sync with the dataset.
===============================================================================
"""

import os
import pickle
import sys

import cv2
import face_recognition

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_DIR = "dataset"
ENCODINGS_FILE = "Encodings.p"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def get_image_files(dataset_dir: str) -> list:
    """
    Return a sorted list of valid image filenames found in `dataset_dir`.

    Raises:
        FileNotFoundError: if the dataset folder does not exist.
        ValueError: if the dataset folder contains no supported images.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset folder '{dataset_dir}' does not exist. "
            f"Please create it and add images named like '101_Rohan.jpg'."
        )

    files = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not files:
        raise ValueError(
            f"No valid images found in '{dataset_dir}'. "
            f"Add images named like '101_Rohan.jpg' and try again."
        )

    return sorted(files)


def parse_filename(filename: str):
    """
    Parse 'RollNo_Name.jpg' -> (roll_no, name).

    Raises:
        ValueError: if the filename does not follow the expected convention.
    """
    name_part = os.path.splitext(filename)[0]  # strip file extension
    parts = name_part.split("_", 1)            # split on the FIRST underscore only

    if len(parts) != 2:
        raise ValueError(
            f"Filename '{filename}' does not follow the 'RollNo_Name' convention."
        )

    roll_no, name = parts[0].strip(), parts[1].strip()

    if not roll_no or not name:
        raise ValueError(f"Filename '{filename}' has an empty Roll No or Name.")

    return roll_no, name


def generate_encodings():
    """Main routine: loops through dataset images, builds encodings, saves pickle."""
    print("=" * 70)
    print("   Smart Attendance System - Face Encoding Generator")
    print("=" * 70)

    # --- Step 1: Validate dataset folder & gather image files ------------- #
    try:
        image_files = get_image_files(DATASET_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    total = len(image_files)
    print(f"[INFO] Found {total} image(s) in '{DATASET_DIR}'. Starting encoding...\n")

    known_encodings = []
    known_roll_nos = []
    known_names = []

    success_count = 0
    failed_files = []

    # --- Step 2: Process every image --------------------------------------- #
    for idx, filename in enumerate(image_files, start=1):
        file_path = os.path.join(DATASET_DIR, filename)
        print(f"[{idx}/{total}] Processing '{filename}' ...", end=" ", flush=True)

        # 2a. Parse Roll No / Name from filename
        try:
            roll_no, name = parse_filename(filename)
        except ValueError as e:
            print("SKIPPED (bad filename)")
            print(f"        -> {e}")
            failed_files.append(filename)
            continue

        # 2b. Read image & compute encoding
        try:
            image_bgr = cv2.imread(file_path)
            if image_bgr is None:
                raise ValueError("Could not read image (corrupted or unsupported format).")

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(image_rgb)
            if len(face_locations) == 0:
                raise ValueError("No face detected in image.")
            if len(face_locations) > 1:
                print(f"(WARNING: {len(face_locations)} faces found, using the first one) ", end="")

            # Only encode the first detected face to keep a 1-to-1 mapping
            face_encodings = face_recognition.face_encodings(
                image_rgb, known_face_locations=[face_locations[0]]
            )
            if len(face_encodings) == 0:
                raise ValueError("Could not compute a face encoding for this image.")

            known_encodings.append(face_encodings[0])
            known_roll_nos.append(roll_no)
            known_names.append(name)

            success_count += 1
            print(f"OK  (Roll No: {roll_no}, Name: {name})")

        except Exception as e:
            print("FAILED")
            print(f"        -> {e}")
            failed_files.append(filename)
            continue

    # --- Step 3: Save the encodings ---------------------------------------- #
    if success_count == 0:
        print("\n[ERROR] No encodings were generated. Aborting save.")
        sys.exit(1)

    encode_data = {
        "encodings": known_encodings,   # list[np.ndarray] - 128-d face encodings
        "roll_nos": known_roll_nos,     # list[str]
        "names": known_names,           # list[str]
    }

    try:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(encode_data, f)
    except Exception as e:
        print(f"[ERROR] Failed to save encodings file: {e}")
        sys.exit(1)

    # --- Step 4: Summary ----------------------------------------------------- #
    print("\n" + "-" * 70)
    print(f"[SUCCESS] Encoded {success_count}/{total} images.")
    if failed_files:
        print(f"[WARNING] {len(failed_files)} file(s) skipped due to errors:")
        for f in failed_files:
            print(f"    - {f}")
    print(f"[INFO] Encodings saved to '{ENCODINGS_FILE}'.")
    print("-" * 70)


if __name__ == "__main__":
    generate_encodings()
