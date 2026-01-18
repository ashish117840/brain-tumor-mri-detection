# Brain Tumor MRI Detection (Flask + TensorFlow)

A simple Flask web app to classify brain MRI images into 4 classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

## Project Structure

- [main.py](main.py) — Flask app (upload + predict)
- [model_loader.py](model_loader.py) — model loading + preprocessing + class mapping
- [templates/index.html](templates/index.html) — UI
- [models/](models/) — model files (NOT included in this repo)
- [sample MRI Images/](sample%20MRI%20Images/) — sample images for testing (optional)

## Setup (Conda)

1. Create / activate environment:

- `conda activate py311`

2. Install dependencies:

- `python -m pip install -r requirements.txt`

## Run

- `python .\main.py`
- Open: `http://127.0.0.1:5000`

## Notes

### TensorFlow startup logs

If you want fewer TensorFlow logs:

- PowerShell: `$env:TF_CPP_MIN_LOG_LEVEL=3; python .\main.py`

### Model files on GitHub (important)

This repository does **not** include trained model files.

Download the model from Google Drive and place it locally:

1. Create the folder: `models/`
2. Put your model inside `models/` (example: `models/mri_vgg16_model.keras`)
3. If your `.keras` file fails to load, the app will fall back to the SavedModel folder:
   `models/mri_vgg16_model_tf-20260113T101522Z-1-001/mri_vgg16_model_tf`

Google Drive link (add yours here):

- https://drive.google.com/file/d/1XFeea2Vtr6WHLTfjmUzmMW4zRRNqKdcA/view?usp=sharing

## Publish to GitHub

From the project folder:

- `git init`
- `git add .`
- `git commit -m "Initial commit"`

Create a new repo on GitHub, then:

- `git branch -M main`
- `git remote add origin https://github.com/ashish117840/brain-tumor-mri-detection.git`
- `git push -u origin main`
