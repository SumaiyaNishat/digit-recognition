# Digit Recognition System

This project detects digits (0-9) from images and also handwritten.

A machine learning–based project that can recognize handwritten digits (0–9) using a **Random Forest Classifier** model.  
It supports **image upload, CSV input, batch processing, and GUI-based interaction**.

---

## Features

- **Image Upload & Preview** – Upload handwritten digit images and see a preview in the GUI.
- **Digit Prediction** – Predicts digits using a trained model with confidence score.
- **Model Training** – Train a Random Forest model on the digits dataset or custom CSV data.
- **CSV & Batch Input** – Supports uploading CSV files or multiple images for predictions.
- **Feature Extraction** – Converts pixel values into features for classification.
- **GUI (Tkinter)** – User-friendly interface for non-technical users.

---

## Technologies Used

- **Python**
- **scikit-learn** – Machine learning model (Random Forest)
- **pandas** – CSV handling
- **numpy** – Array & numerical operations
- **joblib** – Model saving/loading
- **tkinter** – GUI interface
- **matplotlib** – Visualization

---

## Project Structure

```
digit_classifier_csv_ui/
│── data/                 # CSV & datasets
│   └── digit_data.csv
│── model/                # Saved ML models
│── src/
│   ├── gui_app.py        # Main GUI Application
│   └── generate_csv.py   # Script to generate dataset
│── README.md             # Project Documentation
```

---

## How It Works

1. **Generate Dataset**

   ```bash
   python src/generate_csv.py
   ```

   → Creates `digit_data.csv` in the `data/` folder.

2. **Train Model**

   - Model automatically trains on first run.
   - Trained model saved in `model/`.

3. **Run GUI Application**
   ```bash
   python src/gui_app.py
   ```
   - Upload an image → get digit prediction.
   - Upload CSV/Batch Images → get predictions + accuracy.

---
