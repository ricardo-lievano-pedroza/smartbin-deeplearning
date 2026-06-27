# SmartBin — Deep Learning Recycling Classifier

A deep-learning project that classifies waste items into one of six categories — **cardboard, glass, metal, paper, plastic, trash** — and tells the user whether it's recyclable. Built as a final project for a Deep Learning course, featuring a CNN training pipeline and a phone-friendly Streamlit web app.

---

## Project structure

| File | Purpose |
|------|---------|
| `garbage_classification_training.ipynb` | Trains and compares three architectures: baseline CNN, ResNet50 transfer learning, and fine-tuning. Exports the best model. |
| `app.py` | Streamlit frontend — takes a live camera photo or file upload and predicts in real time. Works on iPhone via Safari/Chrome. |
| `garbage_model.keras` | Trained fine-tuned ResNet50 model (tracked via Git LFS). |
| `class_names.json` | Class order, image size, and recyclable/landfill mapping used by the app. |
| `requirements.txt` | Python dependencies for running the app. |
| `pyproject.toml` | Project metadata and dependency spec (used with `uv`). |

---

## Dataset

**Garbage Classification (TrashNet)** — [`asdasdasasdas/garbage-classification`](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) on Kaggle.

- ~2,527 images across 6 classes
- Classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- 80 / 20 train / validation split

---

## Model training

The notebook (`garbage_classification_training.ipynb`) trains and compares three approaches on **Google Colab** (GPU runtime):

### Approach A — Baseline CNN (from scratch)
Three stacked `Conv2D` + `MaxPooling2D` blocks, a `Dense(128)` layer with `Dropout(0.4)`, and a softmax head. Trained with Adam and early stopping — this is the honest baseline.

### Approach B — Transfer learning (frozen ResNet50)
ResNet50 pretrained on ImageNet with all layers frozen. A small custom head (`GlobalAveragePooling2D → Dense(256) → Dropout(0.3) → softmax`) is trained on top. Leverages ImageNet texture features without touching the base weights.

### Approach C — Fine-tuning (deployed model)
The frozen ResNet50 from approach B is unfrozen from layer 140 onward and retrained at a very low learning rate (`1e-5`). This gently adapts the deeper filters to recycling materials while keeping the early, generic layers frozen.

### Training details

| Setting | Value |
|---------|-------|
| Input size | 224 × 224 px |
| Batch size | 32 |
| Augmentation | rotation ±20°, shifts ±10%, horizontal flip, zoom ±20% |
| Overfitting controls | EarlyStopping (patience 6), ReduceLROnPlateau, Dropout, class weights |
| Class imbalance | Balanced class weights (the `trash` class is underrepresented) |
| Metric focus | Per-class F1 + confusion matrix, not just accuracy |

---

## Running the training notebook

1. Open `garbage_classification_training.ipynb` in **Google Colab** with a GPU runtime (`Runtime → Change runtime type → GPU`).
2. Run all cells — the notebook downloads the dataset via `kagglehub` and trains all three models.
3. The final cell saves `garbage_model.keras` and `class_names.json`. Download both files and place them next to `app.py`.

---

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Using it from an iPhone

`st.camera_input()` opens the phone's native camera in Safari/Chrome — no app install needed.

**Option A — Same Wi-Fi (quickest for demos):**
When Streamlit starts it prints a **Network URL** like `http://192.168.x.x:8501`. Type that into the iPhone browser (laptop and phone must be on the same network). Tap **Camera → Take Photo** and the prediction appears instantly.

> iOS requires a secure context (HTTPS) for the camera on most public URLs. On the local Network URL it works over plain HTTP. If the camera is blocked, use option B.

**Option B — Streamlit Community Cloud (HTTPS, shareable):**
1. Push this repo to GitHub (model is tracked via Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set `app.py` as the entry point.
3. You get an `https://…streamlit.app` URL that the camera works on from any phone.

---

## How the app works

1. **Capture** — the user takes a photo via the Camera tab or uploads a file.
2. **Preprocess** — the image is auto-rotated (EXIF), converted to RGB, resized to the model's input size, and rescaled to `[0, 1]` (identical to training).
3. **Predict** — the fine-tuned ResNet50 outputs a probability distribution over the 6 classes.
4. **Display** — the top class drives a colour-coded result card, a recyclable / landfill verdict, and a confidence bar chart over all classes. Low-confidence predictions (< 55%) show a tip to retake the photo.

### Bin mapping

| Class | Bin | Recyclable |
|-------|-----|-----------|
| cardboard | Paper & card | Yes |
| paper | Paper & card | Yes |
| glass | Glass | Yes |
| plastic | Plastics | Yes |
| metal | Metal | Yes |
| trash | General waste | No |

---

## Tech stack

- **TensorFlow / Keras** — model training and inference
- **Streamlit** — web frontend
- **Pillow** — image preprocessing
- **NumPy** — array operations
- **scikit-learn** — class weight computation, evaluation metrics (training notebook only)
- **kagglehub** — dataset download (training notebook only)

---

*Educational MVP. Pair with human oversight before any real recycling decision.*
