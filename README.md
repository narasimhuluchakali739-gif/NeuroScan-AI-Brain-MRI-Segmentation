# NeuroScan AI — Brain MRI Segmentation Research Prototype

A student machine-learning project that uses a U-Net model to segment regions in brain MRI slices. The Streamlit dashboard displays the input MRI, predicted binary mask, and red segmentation overlay, along with pixel measurements and downloadable PDF research reports.

> **Medical safety:** This is an educational research prototype, not a medical diagnostic device. It has not been clinically validated. All results require qualified clinical review. Never use it to diagnose, rule out, or treat a medical condition.

## Features

- Train a compact U-Net on paired MRI images and segmentation masks.
- Resize training images and masks to 128 × 128 pixels.
- Calculate a validation Dice score during training.
- Upload PNG, JPG, JPEG, TIF, and TIFF MRI slices.
- Display the original MRI, predicted binary mask, and segmentation overlay.
- Adjust the threshold used to create the displayed binary mask.
- Review 2D segmented-pixel measurements and download image-inclusive PDF research reports.
- Browse recent scan measurements stored in the current browser session.
- Clearly distinguish segmentation measurements from unavailable clinical conclusions.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Model:** U-Net
- **Web Dashboard:** Streamlit
- **Image Processing:** Pillow and NumPy
- **Reports:** PDF generation library specified in `requirements.txt`

## Dataset

Use the [LGG MRI Segmentation dataset on Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

Review the dataset's current license, terms, and attribution requirements before redistributing data or model weights. Do not commit the dataset to this repository.

After downloading and extracting the dataset, place the `kaggle_3m` directory inside `data/`:

```text
brain_tumor_mri_project/
├── app.py
├── model.py
├── train.py
├── requirements.txt
├── README.md
├── artifacts/
│   └── unet_brain_tumor.pth
└── data/
    └── kaggle_3m/
        └── patient folders and MRI/mask files
```

The training script expects each image to have a matching mask, with `_mask` appended to the image filename stem. For example, `slice_1.tif` should match `slice_1_mask.tif`.

## Setup in VS Code

Python 3.10 or 3.11 is a practical starting point. PyTorch installation requirements depend on your operating system and hardware.

### Windows PowerShell

Run each command separately from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

If PyTorch installation fails, consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the Model

1. Download and extract the dataset into `data/kaggle_3m/`.
2. Activate your virtual environment.
3. Run the training script:

```bash
python train.py
```

The script reports training loss, validation loss, and a batch-averaged validation Dice score. The best checkpoint is saved to:

```text
artifacts/unet_brain_tumor.pth
```

Training time depends on the dataset size and available hardware. For an initial pipeline test, set `EPOCHS = 1` in `train.py`, then increase it after confirming that training works.

## Run the Web App

The application loads the existing checkpoint; it does not automatically download the dataset or train the model.

From the project root, activate the environment and run:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

[http://localhost:8501](http://localhost:8501)

The app preprocesses uploaded slices by converting them to RGB, resizing them to 128 × 128 using bilinear interpolation, and scaling pixel values to [0, 1]. The predicted binary mask is resized to the original image dimensions using nearest-neighbor interpolation for display and pixel counting.

### Understanding the Measurements

The **segmented image area (%)** is calculated as:

`segmented pixels / total pixels × 100`

This is a 2D measurement of the displayed slice. It is not a cancer probability, whole-brain tumor volume, or severity score.

The PDF research report includes the source MRI, predicted mask, overlay, pixel measurements, inference status, clinical-field availability, and limitations.

The model does not provide a calibrated confidence estimate, and no separate validated tumor-classification model is included. Tumor type, grade, stage, and diagnosis cannot be determined by this segmentation model alone.

## GitHub and Deployment

Do not commit the dataset or private patient MRI scans. Check repository file-size limits and the dataset license before publishing model weights.

For Streamlit Community Cloud:

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Connect your GitHub account.
3. Select the `NeuroScan-AI-Brain-MRI-Segmentation` repository.
4. Select `app.py` as the application entrypoint.
5. Deploy the application.

**Important:** The trained checkpoint must be available at `artifacts/unet_brain_tumor.pth` in the deployed environment. If it is not committed to GitHub, configure another permitted way to provide the model before deployment.

See the [official Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Evaluation Limitations

- The starter training pipeline uses a random image-level split. Adjacent slices from the same patient may appear in both training and validation, potentially inflating the validation score.
- Stronger research evaluation should split data by patient ID and test on held-out patients.
- Report metrics such as Dice and Intersection over Union (IoU) on held-out data.
- Any “region marked” message is based on the predicted mask area, not a separately trained tumor/no-tumor classifier.
- Predictions may be incorrect, especially for images that differ from the training dataset.
- This prototype has not been clinically validated and must not be used to make medical decisions.

## Future Improvements

- Patient-level training and validation splits.
- More comprehensive Dice and IoU evaluation.
- Improved visualization and report customization.
- Reproducible experiments and model version tracking.
- Evaluation on an independent, appropriately licensed dataset.

## Author

Developed as a student machine-learning research project.
