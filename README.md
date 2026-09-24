# Smart Attendance System with Face Recognition

A production-ready desktop attendance application built with **Python**,
**CustomTkinter**, **OpenCV**, and **face_recognition**. It recognizes faces
live from a webcam feed and automatically logs attendance to a CSV file.

---

## 1. Folder Structure

```
Smart_Attendance_System/
├── dataset/                 # Put reference photos here: "RollNo_Name.jpg"
│   └── 101_Rohan.jpg        # e.g. Roll No = 101, Name = Rohan
├── attendance.csv           # Auto-created; logs attendance (Roll_No, Name, Date, Time)
├── EncodeGenerator.py       # Step 1: builds face encodings from dataset/
├── Encodings.p              # Auto-generated pickle file with face encodings
├── main.py                  # Step 2: run this to launch the GUI application
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 2. Installation

### Step 2.1 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### Step 2.2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` (a dependency of `face-recognition`) requires a C++
> compiler to build from source. See the troubleshooting section below if
> installation fails — this is the #1 issue Windows users hit.

---

## 3. Usage

### Step 3.1 — Add reference photos

Place one clear, front-facing photo per person inside `dataset/`, named
exactly as:

```
RollNo_Name.jpg
```

Examples: `101_Rohan.jpg`, `102_Priya.jpg`, `103_Amit_Kumar.jpg`
(everything after the first underscore is treated as the name).

### Step 3.2 — Generate face encodings

Run this **once**, and again any time you add/remove photos from `dataset/`:

```bash
python EncodeGenerator.py
```

This creates/updates `Encodings.p`.

### Step 3.3 — Launch the application

```bash
python main.py
```

- Click **Start Camera** to begin the live feed and face recognition.
- Recognized faces are outlined in **green** with a `RollNo - Name` label;
  unrecognized faces are outlined in **red** as `Unknown`.
- The first time a known face is seen on a given day, a row is appended to
  `attendance.csv` and shown in the **Today's Attendance** table. Re-scanning
  the same person the same day will **not** create duplicate entries.
- Click **Stop Camera** to release the webcam.

---

## 4. Troubleshooting: Installing `dlib` / `face-recognition` on Windows

`dlib` compiles native C++ code, so Windows users need build tools and CMake
installed **before** running `pip install`.

### Option A — Install pre-built tools (recommended, easiest)

1. **Install CMake**
   - Download the Windows installer from https://cmake.org/download/
   - During installation, check **"Add CMake to the system PATH"**.
   - Verify: open a new terminal and run `cmake --version`.

2. **Install Visual Studio Build Tools**
   - Download "Build Tools for Visual Studio" from
     https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - In the installer, select the **"Desktop development with C++"** workload.
   - This provides the MSVC compiler that `dlib` needs to build.

3. **Restart your terminal / IDE** so the updated PATH is picked up.

4. **Re-run the install:**
   ```bash
   pip install cmake
   pip install dlib
   pip install face-recognition
   ```

### Option B — Use a pre-compiled dlib wheel (fastest, no compiling)

If compiling still fails, install a matching pre-built wheel for your Python
version instead of building from source:

```bash
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl
```

(Pick the wheel matching your Python version, e.g. `cp310`, `cp311`, `cp312`.)
Then continue with:

```bash
pip install face-recognition
pip install -r requirements.txt
```

### Option C — Use Conda (avoids the compiler step entirely)

```bash
conda create -n attendance python=3.10
conda activate attendance
conda install -c conda-forge dlib
pip install -r requirements.txt
```

### Common errors & fixes

| Error | Fix |
|---|---|
| `CMake is not installed on your system!` | Install CMake and ensure it's on PATH (Option A, step 1). |
| `error: Microsoft Visual C++ 14.0 or greater is required` | Install "Desktop development with C++" workload (Option A, step 2). |
| Build takes forever / hangs | Use a pre-built wheel instead (Option B) or Conda (Option C). |
| `ImportError: DLL load failed` after install | Your Python architecture (32/64-bit) likely doesn't match the wheel — reinstall with the correct one. |

### macOS / Linux

```bash
# macOS (requires Xcode command line tools + cmake via Homebrew)
xcode-select --install
brew install cmake
pip install -r requirements.txt

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake
pip install -r requirements.txt
```

---

## 5. Notes on Accuracy & Performance

- Use clear, well-lit, front-facing photos in `dataset/` for the best
  recognition accuracy.
- `FACE_MATCH_TOLERANCE` in `main.py` (default `0.50`) controls strictness —
  lower it for stricter matching (fewer false positives), raise it if valid
  faces aren't being recognized.
- `PROCESS_EVERY_N_FRAMES` and `FRAME_RESIZE_SCALE` in `main.py` control the
  performance/accuracy trade-off of the live recognition loop; increase the
  frame skip or lower the resize scale on slower machines.
