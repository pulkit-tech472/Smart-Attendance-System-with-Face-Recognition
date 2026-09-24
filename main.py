"""
main.py
===============================================================================
Smart Attendance System with Face Recognition - Main GUI Application
===============================================================================
A modern desktop application (built with CustomTkinter) that:

  1. Loads pre-computed face encodings from `Encodings.p`.
  2. Streams a real-time webcam feed embedded inside the GUI window.
  3. Detects & recognizes faces frame-by-frame using the `face_recognition`
     library (compare_faces + face_distance).
  4. Draws a bounding box + name label around every recognized face.
  5. Automatically logs attendance to `attendance.csv` the first time a
     person is recognized on a given day (no duplicate entries per day).
  6. Displays a live table of everyone marked present today.

Run:
    python main.py

Make sure you have already run `EncodeGenerator.py` at least once so that
`Encodings.p` exists before launching this app.
===============================================================================
"""

import csv
import os
import pickle
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import customtkinter as ctk
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
ENCODINGS_FILE = "Encodings.p"
ATTENDANCE_CSV = "attendance.csv"
CSV_HEADERS = ["Roll_No", "Name", "Date", "Time"]

CAMERA_INDEX = 0                 # default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
VIDEO_LABEL_SIZE = (640, 480)    # display size inside the GUI

FACE_MATCH_TOLERANCE = 0.50      # lower = stricter matching
FRAME_RESIZE_SCALE = 0.25        # downscale factor used for face detection speed
PROCESS_EVERY_N_FRAMES = 2       # run face recognition every Nth frame (perf)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SmartAttendanceApp(ctk.CTk):
    """Main application window for the Smart Attendance System."""

    def __init__(self):
        super().__init__()

        # ---- Window setup -------------------------------------------------- #
        self.title("Smart Attendance System - Face Recognition")
        self.geometry("1100x650")
        self.minsize(950, 600)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ---- Internal state -------------------------------------------------- #
        self.cap = None                 # cv2.VideoCapture instance
        self.camera_running = False     # is the camera loop active?
        self.after_id = None            # id returned by self.after(), used to cancel loop
        self.frame_counter = 0          # used with PROCESS_EVERY_N_FRAMES

        self.known_encodings = []
        self.known_roll_nos = []
        self.known_names = []
        self.encodings_loaded = False

        self.marked_today = set()       # roll numbers already marked present today
        self.today_str = datetime.now().strftime("%Y-%m-%d")

        # Cache of the last computed face boxes/names so we can keep drawing
        # them on frames that are not re-processed (PROCESS_EVERY_N_FRAMES).
        self.last_face_boxes = []       # list of (top, right, bottom, left, label, color)

        # ---- Build UI -------------------------------------------------- #
        self._build_ui()

        # ---- Load resources -------------------------------------------------- #
        self._ensure_csv_exists()
        self._load_today_attendance_into_table()
        self._load_encodings()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        """Builds the full GUI layout: video panel (left) + controls/table (right)."""
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- LEFT PANEL: Video feed ---------------- #
        left_frame = ctk.CTkFrame(self, corner_radius=12)
        left_frame.grid(row=0, column=0, padx=(15, 8), pady=15, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        video_title = ctk.CTkLabel(
            left_frame, text="Live Camera Feed", font=ctk.CTkFont(size=18, weight="bold")
        )
        video_title.grid(row=0, column=0, pady=(12, 5))

        self.video_label = ctk.CTkLabel(
            left_frame,
            text="Camera is OFF\nClick 'Start Camera' to begin",
            width=VIDEO_LABEL_SIZE[0],
            height=VIDEO_LABEL_SIZE[1],
            fg_color="#1a1a1a",
            corner_radius=10,
            font=ctk.CTkFont(size=15),
        )
        self.video_label.grid(row=1, column=0, padx=12, pady=8, sticky="nsew")

        # Camera control buttons
        btn_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, pady=(0, 12))

        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶  Start Camera", width=160, height=38,
            command=self.start_camera, fg_color="#2fa572", hover_color="#26875c"
        )
        self.start_btn.grid(row=0, column=0, padx=8)

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="■  Stop Camera", width=160, height=38,
            command=self.stop_camera, fg_color="#c0392b", hover_color="#992d22",
            state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=8)

        # ---------------- RIGHT PANEL: Status + Table ---------------- #
        right_frame = ctk.CTkFrame(self, corner_radius=12)
        right_frame.grid(row=0, column=1, padx=(8, 15), pady=15, sticky="nsew")
        right_frame.grid_rowconfigure(3, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        status_title = ctk.CTkLabel(
            right_frame, text="System Status", font=ctk.CTkFont(size=18, weight="bold")
        )
        status_title.grid(row=0, column=0, pady=(12, 5), sticky="w", padx=12)

        self.status_label = ctk.CTkLabel(
            right_frame, text="Initializing...", text_color="#f1c40f",
            font=ctk.CTkFont(size=13), anchor="w", justify="left"
        )
        self.status_label.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

        table_title = ctk.CTkLabel(
            right_frame, text=f"Today's Attendance ({self.today_str})",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        table_title.grid(row=2, column=0, pady=(5, 5), sticky="w", padx=12)

        # ---- Attendance table (ttk.Treeview, styled to match dark theme) ---- #
        table_container = ctk.CTkFrame(right_frame, corner_radius=10)
        table_container.grid(row=3, column=0, padx=12, pady=(0, 12), sticky="nsew")
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#2b2b2b", foreground="white",
            fieldbackground="#2b2b2b", rowheight=26, borderwidth=0, font=("Segoe UI", 11)
        )
        style.configure(
            "Treeview.Heading",
            background="#1f6aa5", foreground="white",
            font=("Segoe UI", 11, "bold")
        )
        style.map("Treeview", background=[("selected", "#144870")])

        columns = ("roll_no", "name", "date", "time")
        self.tree = ttk.Treeview(
            table_container, columns=columns, show="headings", style="Treeview"
        )
        self.tree.heading("roll_no", text="Roll No")
        self.tree.heading("name", text="Name")
        self.tree.heading("date", text="Date")
        self.tree.heading("time", text="Time")
        self.tree.column("roll_no", width=70, anchor="center")
        self.tree.column("name", width=120, anchor="center")
        self.tree.column("date", width=90, anchor="center")
        self.tree.column("time", width=90, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=6)

        self.count_label = ctk.CTkLabel(
            right_frame, text="Total marked today: 0", font=ctk.CTkFont(size=13)
        )
        self.count_label.grid(row=4, column=0, pady=(0, 12), sticky="w", padx=12)

    # ------------------------------------------------------------------ #
    # Encodings loading
    # ------------------------------------------------------------------ #
    def _load_encodings(self):
        """Loads Encodings.p into memory. Handles missing/corrupted file gracefully."""
        if not os.path.exists(ENCODINGS_FILE):
            msg = (
                f"'{ENCODINGS_FILE}' not found.\n\n"
                f"Please run 'python EncodeGenerator.py' first to generate "
                f"face encodings from the dataset/ folder."
            )
            self._set_status(msg, color="#e74c3c")
            messagebox.showwarning("Encodings Not Found", msg)
            self.encodings_loaded = False
            return

        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)

            self.known_encodings = data.get("encodings", [])
            self.known_roll_nos = data.get("roll_nos", [])
            self.known_names = data.get("names", [])

            if not self.known_encodings:
                raise ValueError("Encodings file is empty.")

            self.encodings_loaded = True
            self._set_status(
                f"Loaded {len(self.known_encodings)} known face(s).\nReady to start camera.",
                color="#2ecc71"
            )
        except (pickle.UnpicklingError, EOFError, ValueError, AttributeError, KeyError) as e:
            msg = (
                f"Failed to load '{ENCODINGS_FILE}': {e}\n\n"
                f"The file may be corrupted. Try regenerating it with "
                f"'python EncodeGenerator.py'."
            )
            self._set_status(msg, color="#e74c3c")
            messagebox.showerror("Encodings Error", msg)
            self.encodings_loaded = False
        except Exception as e:
            msg = f"Unexpected error loading encodings: {e}"
            self._set_status(msg, color="#e74c3c")
            messagebox.showerror("Encodings Error", msg)
            self.encodings_loaded = False

    # ------------------------------------------------------------------ #
    # CSV / Attendance handling
    # ------------------------------------------------------------------ #
    def _ensure_csv_exists(self):
        """Creates attendance.csv with headers if it doesn't already exist."""
        try:
            if not os.path.exists(ATTENDANCE_CSV) or os.path.getsize(ATTENDANCE_CSV) == 0:
                with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADERS)
        except Exception as e:
            messagebox.showerror(
                "CSV Error", f"Could not create '{ATTENDANCE_CSV}': {e}"
            )

    def _load_today_attendance_into_table(self):
        """
        Reads attendance.csv (if present) and pre-populates the table + the
        `marked_today` set with any entries that already exist for today's
        date. This prevents duplicate marking if the app is restarted.
        """
        if not os.path.exists(ATTENDANCE_CSV):
            return

        try:
            with open(ATTENDANCE_CSV, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if row.get("Date") == self.today_str:
                            roll_no = row.get("Roll_No", "")
                            name = row.get("Name", "")
                            time_str = row.get("Time", "")
                            self.marked_today.add(roll_no)
                            self.tree.insert(
                                "", "end", values=(roll_no, name, self.today_str, time_str)
                            )
                    except Exception:
                        # Skip malformed row but keep processing the rest of the file
                        continue
            self._update_count_label()
        except Exception as e:
            messagebox.showwarning(
                "CSV Read Warning",
                f"Could not fully read existing attendance records: {e}"
            )

    def _mark_attendance(self, roll_no: str, name: str):
        """
        Appends a new attendance row to the CSV (and the live table) the
        first time `roll_no` is recognized today. No-op if already marked.
        """
        if roll_no in self.marked_today:
            return  # already marked today - avoid duplicates

        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")

        try:
            with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([roll_no, name, self.today_str, time_str])
        except Exception as e:
            self._set_status(f"Failed to write attendance for {name}: {e}", color="#e74c3c")
            return

        self.marked_today.add(roll_no)

        # Update table on the main thread (we're already on it since this is
        # only called from the after()-driven frame loop).
        self.tree.insert("", "end", values=(roll_no, name, self.today_str, time_str))
        self.tree.yview_moveto(1.0)  # auto-scroll to latest entry
        self._update_count_label()
        self._set_status(f"✔ Attendance marked: {name} (Roll No {roll_no}) at {time_str}", color="#2ecc71")

    def _update_count_label(self):
        self.count_label.configure(text=f"Total marked today: {len(self.marked_today)}")

    # ------------------------------------------------------------------ #
    # Camera control
    # ------------------------------------------------------------------ #
    def start_camera(self):
        """Opens the webcam and begins the real-time video/recognition loop."""
        if self.camera_running:
            return

        if not self.encodings_loaded:
            proceed = messagebox.askyesno(
                "No Encodings Loaded",
                "No face encodings are loaded, so faces will NOT be recognized "
                "(the camera feed will still work). Continue anyway?"
            )
            if not proceed:
                return

        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                raise IOError(
                    f"Could not access webcam at index {CAMERA_INDEX}. "
                    f"Make sure it is connected and not used by another application."
                )

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self.cap = None
            return

        self.camera_running = True
        self.frame_counter = 0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_status("Camera started. Scanning for faces...", color="#3498db")

        self._update_frame()  # kick off the recursive after() loop

    def stop_camera(self):
        """Stops the video loop and releases the webcam."""
        self.camera_running = False

        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.last_face_boxes = []
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        self.video_label.configure(
            image=None, text="Camera is OFF\nClick 'Start Camera' to begin"
        )
        self._set_status("Camera stopped.", color="#f1c40f")

    # ------------------------------------------------------------------ #
    # Core video / recognition loop
    # ------------------------------------------------------------------ #
    def _update_frame(self):
        """
        Reads one frame from the webcam, (periodically) runs face detection
        & recognition on it, draws overlays, and schedules itself again via
        `self.after()`. This keeps everything on the main GUI thread, which
        avoids cross-thread Tkinter issues.
        """
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._set_status("Warning: Failed to read frame from webcam.", color="#e74c3c")
            self.after_id = self.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)  # mirror for a more natural "selfie" view
        self.frame_counter += 1

        # Only run the (expensive) face recognition every Nth frame for
        # smoother FPS; on skipped frames we simply redraw the last known boxes.
        if self.encodings_loaded and self.frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            try:
                self.last_face_boxes = self._recognize_faces(frame)
            except Exception as e:
                self._set_status(f"Recognition error: {e}", color="#e74c3c")
                self.last_face_boxes = []

        self._draw_overlays(frame, self.last_face_boxes)
        self._render_frame_to_label(frame)

        # Schedule the next frame update (~30 FPS target)
        self.after_id = self.after(15, self._update_frame)

    def _recognize_faces(self, frame):
        """
        Detects faces in `frame`, compares them against known encodings, and
        returns a list of (top, right, bottom, left, label, color) tuples
        scaled back to the full frame size, ready for drawing.
        """
        results = []

        # Downscale for faster detection, then scale coordinates back up
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        import face_recognition  # local import kept alongside cv2 usage for clarity

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        scale = 1.0 / FRAME_RESIZE_SCALE

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = (
                int(top * scale), int(right * scale), int(bottom * scale), int(left * scale)
            )

            label = "Unknown"
            color = (0, 0, 255)  # red (BGR) for unrecognized faces

            if self.known_encodings:
                matches = face_recognition.compare_faces(
                    self.known_encodings, face_encoding, tolerance=FACE_MATCH_TOLERANCE
                )
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        roll_no = self.known_roll_nos[best_match_index]
                        name = self.known_names[best_match_index]
                        label = f"{roll_no} - {name}"
                        color = (0, 200, 0)  # green (BGR) for recognized faces

                        # Log attendance immediately upon recognition
                        self._mark_attendance(roll_no, name)

            results.append((top, right, bottom, left, label, color))

        return results

    @staticmethod
    def _draw_overlays(frame, face_boxes):
        """Draws a sleek rectangle + filled label bar under each detected face."""
        for (top, right, bottom, left, label, color) in face_boxes:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Filled label background for readability
            label_bg_top = max(bottom, 0)
            cv2.rectangle(frame, (left, label_bg_top), (right, label_bg_top + 28), color, cv2.FILLED)
            cv2.putText(
                frame, label, (left + 6, label_bg_top + 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1
            )

    def _render_frame_to_label(self, frame_bgr):
        """Converts an OpenCV BGR frame to a CTkImage and displays it in the video label."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        ctk_image = ctk.CTkImage(
            light_image=pil_image, dark_image=pil_image, size=VIDEO_LABEL_SIZE
        )
        self.video_label.configure(image=ctk_image, text="")
        self.video_label.image = ctk_image  # keep a reference to avoid garbage collection

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def _set_status(self, message: str, color: str = "white"):
        self.status_label.configure(text=message, text_color=color)

    def on_closing(self):
        """Ensures the webcam is released before the window closes."""
        self.stop_camera()
        self.destroy()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        app = SmartAttendanceApp()
        app.mainloop()
    except Exception as fatal_error:
        # Last-resort catch-all so the user always gets a readable error
        # instead of a raw traceback closing an invisible window.
        print(f"[FATAL ERROR] The application failed to start: {fatal_error}")
        try:
            messagebox.showerror("Fatal Error", f"The application failed to start:\n{fatal_error}")
        except Exception:
            pass
