import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
import joblib, os, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageGrab


# ------------------- Paths -------------------
CSV_FILE = "../data/digit_data.csv"
MODEL_FILE = "../models/digit_classifier_model.joblib"
USER_IMG_DIR = "../data/user_images"
DRAW_SAVE_DIR = "../data/saved_drawings"


PIX_COLS = [f"pix{i}" for i in range(64)]


# ------------------- App -------------------
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("820x600")
        self.low_conf_threshold = 80.0  # % below which we ask user to confirm/correct

        self._ensure_dirs()
        self._ensure_dataset_exists()

        # --- Menu ---
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Train Model", command=self.train_model)
        filemenu.add_command(label="Retrain Model (All Data)", command=self.retrain_model)
        filemenu.add_command(label="Predict from CSV", command=self.predict_from_csv_row)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "Digit Recognizer — Advanced\nTkinter + scikit-learn\nFeatures: Draw/Upload, Batch, Confidence, Active Learning"))
        menubar.add_cascade(label="Help", menu=helpmenu)
        root.config(menu=menubar)


        # --- Controls ---
        ctrl = ttk.LabelFrame(root, text="Controls", padding=12)
        ctrl.pack(pady=10, padx=10, fill="x")


        ttk.Button(ctrl, text="Train Model", command=self.train_model).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Retrain Model (All Data)", command=self.retrain_model).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Draw / Upload Digit", command=self.open_draw_window).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Upload Single Image", command=self.upload_single_image).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Upload Batch Images", command=self.upload_batch_images).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Predict from CSV", command=self.predict_from_csv_row).pack(side="left", padx=8)


        # --- Results ---
        res = ttk.LabelFrame(root, text="Results", padding=10)
        res.pack(padx=10, pady=10, fill="both", expand=True)


        self.result_text = tk.Text(res, wrap="word", height=18, state="normal")
        self.result_text.pack(side="left", fill="both", expand=True)


        scrollbar = ttk.Scrollbar(res, command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=scrollbar.set)


        # --- Status ---
        self.status = tk.Label(root, text="Ready", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    # ------------------- Utilities -------------------
    def _ensure_dirs(self):
        os.makedirs("../models", exist_ok=True)
        os.makedirs("../data", exist_ok=True)
        os.makedirs(USER_IMG_DIR, exist_ok=True)
        os.makedirs(DRAW_SAVE_DIR, exist_ok=True)

    def _ensure_dataset_exists(self):
        """Create CSV if missing."""
        if not os.path.exists(CSV_FILE):
            from sklearn.datasets import load_digits
            digits = load_digits()
            df = pd.DataFrame(digits.data, columns=PIX_COLS)
            df["label"] = digits.target
            df.to_csv(CSV_FILE, index=False)
            self._log(f"Dataset not found. Created: {CSV_FILE} ({len(df)} samples)")


    def _log(self, msg):
        self.result_text.insert("end", msg + "\n")
        self.result_text.see("end")


    def _status(self, msg):
        self.status.config(text=msg)

    # --- Preprocessing: convert any PIL image to 8x8 features like sklearn.digits (0..16, background≈0) ---
    def _image_to_features(self, pil_img: Image.Image) -> np.ndarray:
        # Grayscale, improve contrast a bit, resize to 8x8
        img = pil_img.convert("L")
        img = ImageOps.autocontrast(img)
        img = img.resize((8, 8), Image.LANCZOS)

        arr = np.asarray(img, dtype=np.float32)  # 0..255, white background
        # Scale to 0..16 and invert so background ≈ 0, strokes ≈ up to 16
        arr16 = 16.0 - (arr / 255.0 * 16.0)
        # Clip just in case, flatten to 64
        arr16 = np.clip(arr16, 0.0, 16.0).reshape(1, -1)
        return arr16

    def _append_sample(self, features_1x64: np.ndarray, label: int):
        """Append a single labeled example to the CSV dataset."""
        sample = pd.DataFrame(features_1x64, columns=PIX_COLS)
        sample["label"] = label
        # Append without header
        sample.to_csv(CSV_FILE, mode="a", header=False, index=False)

    def _predict_with_conf(self, model, X_1x64: np.ndarray):
        pred = model.predict(X_1x64)[0]
        conf = None
        if hasattr(model, "predict_proba"):
            try:
                conf = float(np.max(model.predict_proba(X_1x64))) * 100.0
            except Exception:
                conf = None
        return pred, conf

    def _load_model(self):
        if not os.path.exists(MODEL_FILE):
            messagebox.showerror("Error", "Model not found. Train the model first.")
            return None
        return joblib.load(MODEL_FILE)


    # ------------------- Train / Retrain -------------------
    def train_model(self):
        if not os.path.exists(CSV_FILE):
            messagebox.showerror("Error", f"CSV not found at {CSV_FILE}")
            return
        df = pd.read_csv(CSV_FILE)
        if "label" not in df.columns or len(df.columns) != 65:
            messagebox.showerror("Error", "CSV format invalid. Expect 64 feature columns + 'label'.")
            return

        X, y = df[PIX_COLS].values, df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_FILE)
        acc = accuracy_score(y_test, model.predict(X_test)) * 100.0

        self._status(f"Model trained (Accuracy: {acc:.2f}%)")
        self._log(f"Model trained successfully! Test Accuracy: {acc:.2f}%  | Samples: {len(df)}")

    def retrain_model(self):
        """Retrain on ALL data (no hold-out) — quick refresh after adding new labeled samples."""
        if not os.path.exists(CSV_FILE):
            messagebox.showerror("Error", f"CSV not found at {CSV_FILE}")
            return
        df = pd.read_csv(CSV_FILE)
        if "label" not in df.columns or len(df.columns) != 65:
            messagebox.showerror("Error", "CSV format invalid. Expect 64 feature columns + 'label'.")
            return

        X, y = df[PIX_COLS].values, df["label"].values
        model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)

        train_acc = model.score(X, y) * 100.0
        self._status(f"🔁 Retrained on {len(y)} samples (Training Acc: {train_acc:.2f}%)")
        self._log(f"Retrained model saved. Training Accuracy: {train_acc:.2f}%")


    # ------------------- Predict from CSV -------------------
    def predict_from_csv_row(self):
        model = self._load_model()
        if model is None:
            return


        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return


        df = pd.read_csv(file_path)
        # If a 'label' column exists, drop it to predict features only.
        features_df = df.drop(columns=["label"], errors="ignore")
        if features_df.shape[1] != 64:
            messagebox.showerror("Error", f"CSV must have 64 feature columns (got {features_df.shape[1]}).")
            return


        preds = model.predict(features_df.values)
        self._log(f"Predictions from {os.path.basename(file_path)} (first 50): {list(preds)[:50]}{'...' if len(preds) > 50 else ''}")
        self._status(f"Predictions complete ({len(preds)} rows)")


    # ------------------- Draw / Upload Window -------------------
    def open_draw_window(self):
        win = tk.Toplevel(self.root)
        win.title("Draw or Upload Digit")
        win.geometry("460x520")


        canvas = tk.Canvas(win, width=320, height=320, bg="white", cursor="cross")
        canvas.pack(pady=10)


        pred_label = tk.Label(win, text="Draw a digit OR upload an image", font=("Arial", 13, "bold"))
        pred_label.pack(pady=6)


        # Drawing handler
        def paint(event):
            r = 10  # brush radius
            x1, y1, x2, y2 = event.x-r, event.y-r, event.x+r, event.y+r
            canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        canvas.bind("<B1-Motion>", paint)


        def clear_canvas():
            canvas.delete("all")
            pred_label.config(text="Draw a digit OR upload an image")


        def save_canvas_png():
            # Save a screenshot of the canvas to PNG
            x = win.winfo_rootx() + canvas.winfo_x()
            y = win.winfo_rooty() + canvas.winfo_y()
            x1, y1 = x + canvas.winfo_width(), y + canvas.winfo_height()
            img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
            os.makedirs(DRAW_SAVE_DIR, exist_ok=True)
            fname = os.path.join(DRAW_SAVE_DIR, f"drawing_{int(time.time())}.png")
            img.save(fname)
            messagebox.showinfo("Saved", f"Drawing saved to:\n{fname}")


        def predict_from_canvas():
            model = self._load_model()
            if model is None:
                return
            # Grab the canvas region
            x = win.winfo_rootx() + canvas.winfo_x()
            y = win.winfo_rooty() + canvas.winfo_y()
            x1, y1 = x + canvas.winfo_width(), y + canvas.winfo_height()
            raw = ImageGrab.grab().crop((x, y, x1, y1))
            feats = self._image_to_features(raw)
            pred, conf = self._predict_with_conf(model, feats)


            # Active learning: confirm/correct if low confidence
            final_label = pred
            if conf is not None and conf < self.low_conf_threshold:
                resp = simpledialog.askstring("Low Confidence",
                                              f"Prediction: {pred} (conf {conf:.2f}%).\n"
                                              f"Press Enter to accept, or type correct digit (0-9):")
                if resp is not None and resp.strip() != "":
                    try:
                        final_label = int(resp)
                    except ValueError:
                        pass  # keep pred


            # Append to dataset (continuous learning)
            try:
                self._append_sample(feats, int(final_label))
                self._log(f"Canvas example saved to dataset with label: {final_label}")
            except Exception as e:
                self._log(f"Could not append canvas sample: {e}")


            if conf is not None:
                pred_label.config(text=f"Canvas Prediction: {pred}  ({conf:.2f}% conf)  | Saved label: {final_label}", fg="green")
            else:
                pred_label.config(text=f"Canvas Prediction: {pred}  | Saved label: {final_label}", fg="green")


        def upload_image_and_predict():
            model = self._load_model()
            if model is None:
                return
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
            )
            if not file_path:
                return


            try:
                pil = Image.open(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot open image: {e}")
                return


            feats = self._image_to_features(pil)
            pred, conf = self._predict_with_conf(model, feats)


            # Active learning: accept or correct
            final_label = pred
            if conf is not None and conf < self.low_conf_threshold:
                resp = simpledialog.askstring("Low Confidence",
                                              f"Prediction: {pred} (conf {conf:.2f}%).\n"
                                              f"Press Enter to accept, or type correct digit (0-9):")
                if resp is not None and resp.strip() != "":
                    try:
                        final_label = int(resp)
                    except ValueError:
                        pass


            # Save a copy of uploaded image and append to dataset
            try:
                os.makedirs(USER_IMG_DIR, exist_ok=True)
                copy_name = os.path.join(USER_IMG_DIR, os.path.basename(file_path))
                pil.convert("L").save(copy_name)
                self._append_sample(feats, int(final_label))
                self._log(f"Uploaded '{os.path.basename(file_path)}' → pred: {pred}"
                          f"{f' ({conf:.2f}% conf)' if conf is not None else ''} | saved label: {final_label}")
            except Exception as e:
                self._log(f"Could not save/append uploaded sample: {e}")


            if conf is not None:
                pred_label.config(text=f"Upload Prediction: {pred}  ({conf:.2f}% conf)  | Saved label: {final_label}", fg="blue")
            else:
                pred_label.config(text=f"Upload Prediction: {pred}  | Saved label: {final_label}", fg="blue")


        # Buttons row
        btns = ttk.Frame(win)
        btns.pack(pady=8)
        ttk.Button(btns, text="Clear", command=clear_canvas).grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(btns, text="Predict Drawing", command=predict_from_canvas).grid(row=0, column=1, padx=6, pady=4)
        ttk.Button(btns, text="Upload Image & Predict", command=upload_image_and_predict).grid(row=0, column=2, padx=6, pady=4)
        ttk.Button(btns, text="Save Drawing as PNG", command=save_canvas_png).grid(row=1, column=0, columnspan=3, pady=6)


    # ------------------- Single / Batch Upload (from main window) -------------------
    def upload_single_image(self):
        model = self._load_model()
        if model is None:
            return
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not file_path:
            return
        self._predict_and_learn_from_file(model, file_path)


    def upload_batch_images(self):
        model = self._load_model()
        if model is None:
            return
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not file_paths:
            return
        for p in file_paths:
            self._predict_and_learn_from_file(model, p)

    def _predict_and_learn_from_file(self, model, file_path: str):
        try:
            pil = Image.open(file_path)
        except Exception as e:
            self._log(f"Cannot open image '{file_path}': {e}")
            return

        feats = self._image_to_features(pil)
        pred, conf = self._predict_with_conf(model, feats)


        # Ask only when low confidence; otherwise auto-accept
        final_label = pred
        if conf is not None and conf < self.low_conf_threshold:
            resp = simpledialog.askstring("Low Confidence",
                                          f"{os.path.basename(file_path)}\nPrediction: {pred} (conf {conf:.2f}%).\n"
                                          f"Press Enter to accept, or type correct digit (0-9):")
            if resp is not None and resp.strip() != "":
                try:
                    final_label = int(resp)
                except ValueError:
                    pass


        # Save copy & append
        try:
            os.makedirs(USER_IMG_DIR, exist_ok=True)
            copy_name = os.path.join(USER_IMG_DIR, os.path.basename(file_path))
            pil.convert("L").save(copy_name)
            self._append_sample(feats, int(final_label))
            self._log(f"{os.path.basename(file_path)} → pred: {pred}"
                      f"{f' ({conf:.2f}% conf)' if conf is not None else ''} | saved label: {final_label}")
            self._status(f"✅ Predicted: {pred}")
        except Exception as e:
            self._log(f"Could not save/append sample for '{file_path}': {e}")


# ------------------- Run -------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()


