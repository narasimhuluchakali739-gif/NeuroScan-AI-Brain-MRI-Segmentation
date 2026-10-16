# NeuroScan AI — Brain MRI Segmentation Research Prototype

A student machine-learning project that uses a U-Net model to mark regions in brain MRI slices. The Streamlit research dashboard displays the input MRI, predicted binary mask, and a red overlay, with pixel-count measurements and downloadable PDF research reports.

> **Medical safety:** This is an educational research prototype, not a medical diagnostic device. It has not been clinically validated. All results require qualified clinical review. Never use it to diagnose, rule out, or treat a medical condition.

## Features

- Train a compact U-Net on paired MRI images and segmentation masks.
- Preprocess training images and masks to 128 × 128 pixels.
- Report a validation Dice score during training.
- Upload PNG, JPG, JPEG, TIF, or TIFF MRI slices in the Streamlit app.
- Display the original image, predicted binary mask, and segmentation overlay.
- Adjust the mask threshold used to construct the displayed binary mask.
- Review 2D segmented-pixel measurements and download an image-inclusive PDF research report.
- Browse recent scan measurements stored in the current browser session.
- Distinguish measured segmentation output from unavailable classification and clinical conclusions.

## Dataset

Use the **LGG MRI Segmentation** dataset published on Kaggle:
[LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

Review the dataset's current license, terms, and attribution requirements before redistributing data or model weights. Do not commit the dataset to this repository.

After downloading and extracting, place the `kaggle_3m` directory here:

```text
brain-tumor-mri-project/
└── data/
    └── kaggle_3m/
        ├── TCGA_.../
        │   ├── ..._1.tif
        │   ├── ..._1_mask.tif
        │   └── ...
        └── ...
```

The training script expects each image to have a matching mask named by appending `_mask` to the image stem, for example `slice_1.tif` and `slice_1_mask.tif`.

## Setup in VS Code

Python 3.10 or 3.11 is a practical starting point. A compatible PyTorch build can depend on your operating system and whether you have a supported GPU.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, install the build recommended by the official PyTorch selector: [PyTorch installation guide](https://pytorch.org/get-started/locally/)

## Train the model

1. Download and extract the dataset into `data/kaggle_3m/`.
2. From the project root, run:

```bash
python train.py
```

1. The script prints training loss, validation loss, and a batch-averaged validation Dice score.
1. The best checkpoint is saved to `artifacts/unet_brain_tumor.pth`.

Training time depends on the number of images and available hardware. For a first smoke test, set `EPOCHS = 1` in `train.py`; then increase it after the full pipeline works.

## Run the web app

The app uses the existing checkpoint at `artifacts/unet_brain_tumor.pth`. It does not download data or train automatically. From the project root, activate the environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

For Command Prompt, activate with `.venv\Scripts\activate.bat` before running `python -m streamlit run app.py`.

The app matches training preprocessing: convert the uploaded slice to RGB, resize it to 128 × 128 with bilinear interpolation, and scale pixel values to [0, 1]. The resulting binary mask is enlarged to the source image size using nearest-neighbor interpolation for display and 2D pixel counting.

The reported **segmented image area (%)** is segmented pixels divided by total pixels in that displayed slice, multiplied by 100. It is not a cancer probability, whole-brain tumor volume, or severity score. Research PDFs include the source MRI, predicted mask, overlay, pixel measurements, inference status, clinical-field availability, and limitations. The model has no calibrated confidence estimate, and no separate validated tumor classification model is included. Tumor type, grade, stage, and diagnosis cannot be determined by this segmentation model alone.

## GitHub and deployment

Do not commit the dataset. Model weights may be large; check repository file-size limits and the dataset license before deciding whether to publish a checkpoint. If the checkpoint is not in the Git repository, the app needs another permitted way to access it before deployment.

For Streamlit Community Cloud, connect the GitHub repository, choose `app.py` as the entrypoint, and deploy. The app will not work until the trained checkpoint is available at `artifacts/unet_brain_tumor.pth` in the deployed environment. See:
[Streamlit Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Evaluation limitations

- This starter uses a random image-level split. Adjacent slices from the same patient may appear in both training and validation, which can inflate the validation score.
- For a stronger research project, split by patient ID, evaluate on held-out patients, report Dice and IoU, and document limitations.
- The app's “region marked” message is based on mask area, not a separate trained tumor/no-tumor classifier.
- Predictions may be wrong, especially on images unlike the training data.
