import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from gesture_engine import GestureEngine
import time
from tkinter import messagebox
import os
import threading

# --- UI and Video Parameters ---
VIDEO_WIDTH = 960  
VIDEO_HEIGHT = 540
WINDOW_WIDTH = VIDEO_WIDTH+50
WINDOW_HEIGHT = VIDEO_HEIGHT+150
FRAME_INTERVAL = 5  # in milliseconds

def api_call(sentence):
    time.sleep(2)
    print(f"[API] Finished processing: {sentence}")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition GUI")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.attributes("-fullscreen", False)

        self.last_frame_time = time.time()

        # Text Display (top bar)
        self.text_label = tk.Label(root, text="", font=("Helvetica", 14), fg="white", bg="black", height=2)
        self.text_label.pack(fill=tk.X)

        # Video Display
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=5)

        # Control Buttons
        controls = ttk.Frame(root)
        controls.pack(pady=10)

        self.start_btn = ttk.Button(controls, text="Start Recognition", command=self.toggle_recognition)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.draw_btn = ttk.Button(controls, text="Toggle Drawing", command=self.toggle_drawing)
        self.draw_btn.pack(side=tk.LEFT, padx=5)

        self.quit_btn = ttk.Button(controls, text="Quit", command=self.quit_app)
        self.quit_btn.pack(side=tk.LEFT, padx=5)

        # Camera and Engine
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam. Make sure it's not in use by another app.")
            self.root.destroy() 
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        self.engine = GestureEngine()
        self.recognition_active = False
        self.last_sentence = ""
        self.api_thread_running = False

        # Start video loop
        self.update_video()

    def toggle_recognition(self):
        self.recognition_active = not self.recognition_active
        state = "ON" if self.recognition_active else "OFF"
        print(f"[App] Recognition toggled: {state}")
        self.start_btn.config(text="Stop Recognition" if self.recognition_active else "Start Recognition")

    def toggle_drawing(self):
        self.engine.toggle_draw()

    def update_fps(self, frame):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def maybe_call_api(self, sentence):
        if sentence != self.last_sentence and sentence.strip():
            self.last_sentence = sentence
            if not self.api_thread_running:
                self.api_thread_running = True
                threading.Thread(target=self.run_api_call, args=(sentence,), daemon=True).start()

    def run_api_call(self, sentence):
        api_call(sentence)
        self.api_thread_running = False

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

            if self.recognition_active:
                frame, sentence = self.engine.process(frame)
                self.text_label.config(text=sentence)
                self.maybe_call_api(sentence)
            else:
                sentence = "Recognition is OFF"
                self.text_label.config(text=sentence)

            frame = cv2.flip(frame, 1)
            self.update_fps(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.root.after(FRAME_INTERVAL, self.update_video)

    def quit_app(self):
        print("[App] Exiting...")
        self.cap.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()