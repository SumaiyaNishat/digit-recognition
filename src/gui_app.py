import os
import sys
import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageGrab
import matplotlib.pyplot as plt
import customtkinter as ctk
from tkinter import filedialog, messagebox

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_utils import train_model, load_model, predict_single, predict_batch

PREDICTION_LOG = "outputs/predictions.csv"

# --- GUI Appearance Settings ---
CANVAS_WIDTH = 300
CANVAS_HEIGHT = 300
CANVAS_BG = "white"
BUTTON_WIDTH = 180
BUTTON_HEIGHT = 40
FONT_LARGE = ("Arial", 18)
HISTORY_WIDTH = 550
HISTORY_HEIGHT = 600
APP_WIDTH = 1000
APP_HEIGHT = 700
CTK_THEME = "Dark"
CTK_COLOR_THEME = "blue"

ctk.set_appearance_mode(CTK_THEME)
ctk.set_default_color_theme(CTK_COLOR_THEME)


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer App")
        self.root.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.history = []

        # Load or train model
        try:
            self.model = load_model()
        except:
            self.model, acc, _ = train_model()
            messagebox.showinfo("Training", f"Model trained. Accuracy: {acc:.2%}")

        self._build_gui()

    def _build_gui(self):
        # Frames
        left_frame = ctk.CTkFrame(self.root, width=350, corner_radius=10)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)

        right_frame = ctk.CTkFrame(self.root, corner_radius=10)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Canvas for drawing
        self.canvas = ctk.CTkCanvas(left_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg=CANVAS_BG)
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons
        ctk.CTkButton(left_frame, text="Predict Drawing", width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                      command=self.predict_drawing).pack(pady=5)
        ctk.CTkButton(left_frame, text="Clear Canvas", width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                      command=self.clear_canvas).pack(pady=5)
        ctk.CTkButton(left_frame, text="Upload Single Image", width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                      command=self.upload_single_image).pack(pady=5)
        ctk.CTkButton(left_frame, text="Upload Batch Images", width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                      command=self.upload_batch_images).pack(pady=5)
        ctk.CTkButton(left_frame, text="Retrain Model", width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                      command=self.retrain_model).pack(pady=5)
        ctk.CTkButton(left_frame, text="Show Confusion Matrix", width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                      command=self.show_confusion_matrix).pack(pady=5)

        # Prediction Result
        self.result_label = ctk.CTkLabel(left_frame, text="Prediction: ", font=FONT_LARGE)
        self.result_label.pack(pady=10)

        # History
        history_label = ctk.CTkLabel(right_frame, text="Prediction History", font=FONT_LARGE)
        history_label.pack(pady=5)

        self.history_box = ctk.CTkTextbox(right_frame, width=HISTORY_WIDTH, height=HISTORY_HEIGHT)
        self.history_box.pack(padx=5, pady=5, fill="both", expand=True)
        self.history_box.configure(state="disabled")

    # --- Canvas Functions ---
    def draw(self, event):
        r = 8
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    # --- Image Preprocessing ---
    def preprocess_image(self, img):
        img = img.convert("L")  # grayscale
        img = ImageOps.invert(img)  # invert
        img = img.resize((8, 8))  # resize like sklearn digits
        arr = np.array(img)
        arr = (arr / 255.0) * 16  # scale
        return arr.flatten()

    # --- Prediction Functions ---
    def predict_drawing(self):
        self.root.update()
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab(bbox=(x, y, x1, y1))
        features = self.preprocess_image(img)
        pred, conf = predict_single(self.model, features)
        self.show_prediction("Drawing", pred, conf)

    def upload_single_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not filepath:
            return
        img = Image.open(filepath)
        features = self.preprocess_image(img)
        pred, conf = predict_single(self.model, features)
        self.show_prediction(os.path.basename(filepath), pred, conf)

    def upload_batch_images(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not filepaths:
            return
        X = []
        names = []
        for fp in filepaths:
            img = Image.open(fp)
            features = self.preprocess_image(img)
            X.append(features)
            names.append(os.path.basename(fp))
        preds, probs = predict_batch(self.model, np.array(X))
        for name, p, pr in zip(names, preds, probs):
            self.show_prediction(name, p, pr)

    def retrain_model(self):
        self.model, acc, _ = train_model()
        messagebox.showinfo("Retrain", f"Model retrained. Accuracy: {acc:.2%}")

    def show_confusion_matrix(self):
        _, acc, cm = train_model()
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix (Acc {acc:.2%})")
        plt.colorbar()
        plt.show()

    def show_prediction(self, source, pred, conf):
        text = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {source} → Pred: {pred} (Conf {conf:.2%})"
        self.result_label.configure(text=f"Prediction: {pred} ({conf:.2%})")

        self.history_box.configure(state="normal")
        self.history_box.insert("end", text + "\n")
        self.history_box.see("end")
        self.history_box.configure(state="disabled")

        self.save_prediction_log(source, pred, conf)

    def save_prediction_log(self, source, pred, conf):
        os.makedirs("outputs", exist_ok=True)
        log_entry = {"time": datetime.datetime.now(), "source": source, "pred": pred, "confidence": conf}
        if os.path.exists(PREDICTION_LOG):
            df = pd.read_csv(PREDICTION_LOG)
            df = pd.concat([df, pd.DataFrame([log_entry])])
        else:
            df = pd.DataFrame([log_entry])
        df.to_csv(PREDICTION_LOG, index=False)


if __name__ == "__main__":
    root = ctk.CTk()
    app = DigitRecognizerApp(root)
    root.mainloop()
