# Smart-Attendance-System-with-Face-Recognition
# 🎯 Smart Attendance System with Face Recognition

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)

A modern, automated Desktop Application designed to mark attendance in real-time using Face Recognition technology. Built with Python, OpenCV, `face_recognition`, and CustomTkinter.

---

## ✨ Key Features

- **⚡ Real-Time Face Recognition:** Detects and identifies registered faces instantly from a live webcam stream.
- **📁 Automated Dataset Encoding:** Pre-processes image datasets into facial encodings (`.p` pickle file) for fast performance.
- **📊 Duplicate Entry Prevention:** Logically prevents marking duplicate attendance for the same person on the same date.
- **📄 CSV Log Export:** Automatically generates and updates `attendance.csv` with fields: `Roll_No`, `Name`, `Date`, and `Time`.
- **🎨 Modern GUI Dashboard:** Built using `CustomTkinter` featuring a dark mode aesthetic, live webcam view, control buttons, and a real-time attendance table.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.9+
- **GUI Framework:** CustomTkinter / Tkinter
- **Computer Vision & ML:** OpenCV (`cv2`), `face_recognition`, `dlib`
- **Data Handling:** NumPy, Pandas, Pickle

---

## 📂 Project Structure

```text
Smart_Attendance_System/
│
├── dataset/              # Stores images named as "RollNo_Name.jpg"
├── attendance.csv        # Auto-generated CSV file for attendance logs
├── EncodeGenerator.py    # Script to extract and save facial encodings
├── Encodings.p           # Generated pickle binary file
├── main.py               # Main CustomTkinter GUI & Recognition app
└── requirements.txt      # Project dependencies list
