from datetime import datetime
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import streamlit as st
import torch

from model import UNet

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "unet_brain_tumor.pth"
IMAGE_SIZE = 128
DEVICE = torch.device("cpu")
ACCEPTED_FORMATS = ["png", "jpg", "jpeg", "tif", "tiff"]
UNAVAILABLE = (
    "Not available — requires a validated classification model and/or specialist assessment"
)
DISCLAIMER = (
    "EDUCATIONAL RESEARCH ONLY — NOT A MEDICAL DIAGNOSTIC DEVICE. "
    "Do not use this result to diagnose, rule out, or treat a medical condition."
)

st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Segmentation",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ns-bg: #07111f;
        --ns-panel: #0d1b2c;
        --ns-panel-2: #102238;
        --ns-cyan: #54d8e8;
        --ns-muted: #8da4ba;
        --ns-line: rgba(142, 184, 204, .16);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% -15%, rgba(25, 116, 144, .20), transparent 42%),
            linear-gradient(150deg, #07111f 0%, #091524 58%, #07111f 100%);
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .82); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1929 0%, #081321 100%);
        border-right: 1px solid var(--ns-line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #a9bdce; }
    .ns-brand {
        padding: .3rem 0 1.25rem;
        border-bottom: 1px solid var(--ns-line);
        margin-bottom: 1.2rem;
    }
    .ns-brand-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 2.25rem; height: 2.25rem; margin-right: .6rem;
        border: 1px solid rgba(84, 216, 232, .4); border-radius: .75rem;
        color: var(--ns-cyan); background: rgba(84, 216, 232, .08);
        font-size: 1.25rem; vertical-align: middle;
    }
    .ns-brand-name {
        color: #effaff; font-size: 1.08rem; font-weight: 700;
        letter-spacing: .02em; vertical-align: middle;
    }
    .ns-brand-sub {
        display: block; margin: .55rem 0 0 2.9rem;
        color: var(--ns-muted); font-size: .7rem; letter-spacing: .12em;
        text-transform: uppercase;
    }
    .ns-eyebrow {
        color: var(--ns-cyan); font-size: .72rem; letter-spacing: .15em;
        text-transform: uppercase; font-weight: 700; margin-bottom: .35rem;
    }
    .ns-subtle, .ns-viewer-note { color: var(--ns-muted); }
    .ns-viewer-note { font-size: .78rem; padding-top: .35rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(16, 34, 56, .94), rgba(11, 26, 43, .94));
        border: 1px solid var(--ns-line); border-radius: 14px; padding: 1rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] p { color: #9db4c8; }
    div[data-testid="stMetricValue"] { color: #e9fbff; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--ns-line);
        border-radius: 14px;
    }
    [data-testid="stFileUploader"] {
        background: rgba(12, 29, 47, .72);
        border: 1px dashed rgba(84, 216, 232, .32);
        border-radius: 14px;
        padding: .45rem;
    }
    [data-testid="stImage"] img { max-width: 100%; height: auto; }
    @media (max-width: 760px) {
        .ns-brand-sub { letter-spacing: .07em; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {MODEL_PATH}. Add the existing model "
            "there; the app does not download data or train a model automatically."
        )

    try:
        model = UNet()
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise RuntimeError(f"Could not load trained checkpoint at {MODEL_PATH}: {exc}") from exc
    return model


def prepare_image(pil_image):
    """Match the training dataset's RGB, bilinear, [0, 1] 128px preprocessing."""
    resized = pil_image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def make_overlay(image, mask):
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    highlight = np.zeros_like(base)
    highlight[..., 0] = 255
    highlight[..., 1] = 72
    alpha = 0.48
    base[mask] = (base[mask] * (1 - alpha) + highlight[mask] * alpha).astype(np.uint8)
    return Image.fromarray(base)


def image_as_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def report_font(size, bold=False):
    windows_dir = os.environ.get("WINDIR")
    if windows_dir:
        font_path = Path(windows_dir) / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf")
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def draw_wrapped_text(draw, text, x, y, max_width, font, fill, line_gap=7):
    lines = []
    line = ""
    for word in str(text).split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + line_gap
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def add_report_section(draw, title, y):
    draw.text((82, y), title, font=report_font(30, bold=True), fill="#54D8E8")
    return y + 52


def make_pdf_report(scan):
    width, height = 1240, 1754
    background = "#091625"
    text_color = "#EAF5FB"
    muted = "#A9BDCE"
    cyan = "#54D8E8"
    page = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(page)

    draw.rounded_rectangle((58, 54, width - 58, 188), radius=24, fill="#102238")
    draw.text((88, 78), "NEUROSCAN AI", font=report_font(27, bold=True), fill=cyan)
    draw.text(
        (88, 115),
        "Brain MRI segmentation · research report",
        font=report_font(34, bold=True),
        fill=text_color,
    )
    draw.text(
        (88, 159),
        f"Generated {scan['timestamp']}",
        font=report_font(19),
        fill=muted,
    )

    draw.rounded_rectangle(
        (58, 216, width - 58, 366),
        radius=18,
        fill="#321F2A",
        outline="#D77A74",
        width=2,
    )
    draw.text((84, 239), "EDUCATIONAL RESEARCH ONLY", font=report_font(25, bold=True), fill="#FFB4A8")
    draw_wrapped_text(
        draw,
        "Not a medical diagnostic device. Do not use this report to diagnose, rule out, "
        "or treat a medical condition. All results require qualified clinical review.",
        84,
        280,
        width - 168,
        report_font(20),
        text_color,
    )

    y = add_report_section(draw, "Study details", 406)
    details = [
        ("Uploaded image", scan["filename"]),
        ("Image dimensions", f"{scan['width']} × {scan['height']} pixels"),
        ("Model checkpoint", "artifacts/unet_brain_tumor.pth"),
        ("Model status", scan["model_status"]),
        ("Inference status", scan["status"]),
    ]
    for label, value in details:
        draw.text((86, y), label, font=report_font(20, bold=True), fill=muted)
        y = draw_wrapped_text(
            draw,
            value,
            360,
            y,
            width - 446,
            report_font(20),
            text_color,
            line_gap=4,
        )
        y += 10

    y = add_report_section(draw, "Segmentation measurements", y + 12)
    measurements = [
        f"Segmented image area: {scan['percentage']:.4f}%",
        f"Segmented pixels: {scan['segmented_pixels']:,}",
        f"Total image pixels: {scan['total_pixels']:,}",
        f"Segmentation threshold: {scan['threshold']:.2f}",
    ]
    for measurement in measurements:
        draw.text((88, y), "•", font=report_font(23, bold=True), fill=cyan)
        draw.text((122, y), measurement, font=report_font(22), fill=text_color)
        y += 45
    y = draw_wrapped_text(
        draw,
        "Segmented image area is the share of pixels marked in this single 2D image. "
        "It is not tumor volume, cancer probability, or a severity score.",
        88,
        y + 5,
        width - 176,
        report_font(19),
        muted,
    )

    y = add_report_section(draw, "Clinical classification availability", y + 18)
    clinical_fields = [
        ("Tumor type", UNAVAILABLE),
        ("Tumor grade", UNAVAILABLE),
        ("Clinical stage", UNAVAILABLE),
        ("Model confidence", UNAVAILABLE),
        ("Clinical review status", "Not reviewed — specialist assessment required"),
    ]
    for label, value in clinical_fields:
        draw.text((88, y), label, font=report_font(19, bold=True), fill=muted)
        y = draw_wrapped_text(
            draw,
            value,
            360,
            y,
            width - 446,
            report_font(18),
            text_color,
            line_gap=3,
        )
        y += 9

    y += 5
    draw_wrapped_text(
        draw,
        "Limitations: This is a binary 2D segmentation output. No validated classification "
        "model, calibrated confidence measure, diagnosis, grade, or stage is available. "
        "Image acquisition details and clinical context are not assessed.",
        88,
        y,
        width - 176,
        report_font(17),
        muted,
    )

    image_page = Image.new("RGB", (width, height), background)
    image_draw = ImageDraw.Draw(image_page)
    image_draw.text(
        (78, 60),
        "Segmentation views",
        font=report_font(36, bold=True),
        fill=text_color,
    )
    image_draw.text(
        (80, 112),
        f"{scan['filename']} · {scan['width']} × {scan['height']} px",
        font=report_font(19),
        fill=muted,
    )
    panels = [
        ("Original MRI", scan["original_png"]),
        ("Predicted segmentation mask", scan["mask_png"]),
        ("MRI segmentation overlay", scan["overlay_png"]),
    ]
    y = 164
    for title, image_bytes in panels:
        image_draw.rounded_rectangle(
            (58, y, width - 58, y + 482),
            radius=18,
            fill="#102238",
            outline="#284154",
            width=2,
        )
        image_draw.text((82, y + 20), title, font=report_font(24, bold=True), fill=cyan)
        with Image.open(BytesIO(image_bytes)) as source_image:
            panel_image = source_image.convert("RGB")
            panel_image.thumbnail((width - 188, 415), Image.Resampling.LANCZOS)
        image_x = (width - panel_image.width) // 2
        image_page.paste(panel_image, (image_x, y + 56))
        y += 510

    output = BytesIO()
    page.save(
        output,
        format="PDF",
        save_all=True,
        append_images=[image_page],
        resolution=150.0,
    )
    return output.getvalue()


def render_header():
    st.markdown(
        '<div class="ns-eyebrow">Neuroimaging · Research interface</div>',
        unsafe_allow_html=True,
    )
    st.title("NeuroScan AI — Brain MRI Segmentation")
    st.markdown(
        '<p class="ns-subtle">A focused workspace for reviewing model-generated 2D '
        "segmentation alongside the source MRI. Segmentation output is not a clinical "
        "interpretation.</p>",
        unsafe_allow_html=True,
    )
    st.warning(DISCLAIMER, icon=":material/health_and_safety:")


def render_classification_panel():
    st.markdown("#### Classification and clinical information")
    st.caption(
        "These clinical conclusions are intentionally not inferred from a segmentation mask."
    )
    fields = [
        ("Tumor type", UNAVAILABLE),
        ("Tumor grade", UNAVAILABLE),
        ("Clinical stage", UNAVAILABLE),
        ("Model confidence", UNAVAILABLE),
        ("Clinical review status", "Not reviewed — specialist assessment required"),
    ]
    columns = st.columns(2)
    for index, (label, value) in enumerate(fields):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.caption(value)


def render_report_download(scan, key):
    st.download_button(
        "Download research report (PDF)",
        data=make_pdf_report(scan),
        file_name=f"neuroscan_research_report_{scan['id'][:8]}.pdf",
        mime="application/pdf",
        key=key,
        icon=":material/download:",
        width="stretch",
    )


with st.sidebar:
    st.markdown(
        """
        <div class="ns-brand">
            <span class="ns-brand-mark">✳</span>
            <span class="ns-brand-name">NeuroScan AI</span>
            <span class="ns-brand-sub">Imaging research workspace</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Workspace**")
    page = st.radio(
        "Navigation",
        ["Dashboard", "MRI Analysis", "Research Reports", "Model Information"],
        label_visibility="collapsed",
        key="navigation",
    )
    st.markdown("---")
    st.caption("MODEL")
    st.markdown("Compact U-Net · binary segmentation")
    st.caption("INPUT")
    st.markdown("Single 2D MRI slice · RGB · 128 × 128")
    st.markdown("---")
    st.caption("Educational research prototype only. Not a medical diagnostic device.")

render_header()

if page == "Dashboard":
    st.subheader("Research dashboard")
    recent_scans = st.session_state.get("recent_scans", [])
    latest_scan = st.session_state.get("latest_scan")
    with st.container(horizontal=True):
        st.metric("Analyses this session", len(recent_scans), border=True)
        st.metric(
            "Trained checkpoint",
            "Available" if MODEL_PATH.is_file() else "Missing",
            border=True,
        )
        st.metric(
            "Latest segmentation",
            f"{latest_scan['percentage']:.2f}%" if latest_scan else "Not available",
            border=True,
        )
    st.caption(
        "Segmented image area (%) describes marked pixels in one 2D image only; "
        "it is not tumor volume, cancer probability, or severity."
    )
    dashboard_columns = st.columns([1.35, 1])
    with dashboard_columns[0]:
        with st.container(border=True):
            st.markdown("**Analysis workspace**")
            st.write(
                "Upload a PNG, JPG, JPEG, TIF, or TIFF slice to compare the source MRI, "
                "U-Net mask, and color overlay with quantitative pixel measurements."
            )
            st.info(
                "Tumor type, grade, stage, and model confidence are not available from "
                "this segmentation model."
            )
    with dashboard_columns[1]:
        with st.container(border=True):
            st.markdown("**Latest analysis**")
            if latest_scan:
                st.markdown(f"**{latest_scan['filename']}**")
                st.caption(
                    f"{latest_scan['width']} × {latest_scan['height']} px · "
                    f"{latest_scan['segmented_pixels']:,} segmented pixels · "
                    f"{latest_scan['percentage']:.2f}% segmented image area"
                )
                st.badge(latest_scan["status"], color="green")
            else:
                st.caption("Complete an MRI analysis to populate this panel.")

    st.subheader("Recent research activity")
    if not recent_scans:
        st.info("No scans have been analyzed in this browser session.")
    else:
        for scan in recent_scans:
            with st.container(border=True):
                st.markdown(f"**{scan['filename']}**")
                st.caption(
                    f"{scan['timestamp']} · {scan['width']} × {scan['height']} px · "
                    f"{scan['segmented_pixels']:,} / {scan['total_pixels']:,} pixels · "
                    f"{scan['percentage']:.2f}% segmented image area"
                )

elif page == "MRI Analysis":
    st.subheader("MRI analysis workspace")
    st.markdown(
        "Upload one 2D slice. The model applies the same RGB conversion, 128 × 128 "
        "bilinear resize, and [0, 1] normalization used during training."
    )
    uploaded = st.file_uploader(
        "Upload an MRI slice",
        type=ACCEPTED_FORMATS,
        help="Supported formats: PNG, JPG, JPEG, TIF, and TIFF. One 2D slice is analyzed.",
    )
    threshold = st.slider(
        "Segmentation threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help=(
            "Pixels whose sigmoid output meets this threshold are included in the binary "
            "mask. This is not a confidence or diagnosis threshold."
        ),
    )
    st.caption(f"Current mask threshold: {threshold:.2f}")

    if uploaded is None:
        st.info("Upload a compatible 2D MRI image to begin segmentation.")
        st.stop()

    try:
        uploaded_bytes = uploaded.getvalue()
        with Image.open(BytesIO(uploaded_bytes)) as opened_image:
            opened_image.seek(0)
            original = opened_image.convert("RGB")
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        st.error(
            "Unable to read this image. Upload a valid 2D PNG, JPG, JPEG, TIF, or TIFF "
            f"file. Details: {exc}"
        )
        st.stop()

    if original.width < 1 or original.height < 1:
        st.error("The uploaded image has invalid dimensions.")
        st.stop()

    scan_id = sha256(uploaded_bytes).hexdigest()
    try:
        with st.spinner("Loading the trained U-Net and generating the segmentation…"):
            model = load_model()
            if st.session_state.get("segmentation_scan_id") == scan_id:
                probabilities = st.session_state["segmentation_probabilities"]
            else:
                input_tensor = prepare_image(original)
                with torch.inference_mode():
                    logits = model(input_tensor.to(DEVICE))
                    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
                        raise ValueError(
                            "The model returned an unexpected output shape; "
                            "expected one binary mask."
                        )
                    probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise ValueError("The model output contains invalid numeric values.")
                st.session_state["segmentation_scan_id"] = scan_id
                st.session_state["segmentation_probabilities"] = probabilities
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"A valid segmentation was not generated. {exc}")
        st.stop()

    mask_at_model_size = probabilities >= threshold
    mask_image = Image.fromarray(mask_at_model_size.astype(np.uint8) * 255)
    mask_image = mask_image.resize(original.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_image, dtype=np.uint8) > 0
    segmented_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    percentage = segmented_pixels / total_pixels * 100
    overlay = make_overlay(original, mask)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    status = (
        "Inference completed — mask generated"
        if segmented_pixels
        else "Inference completed — no pixels met the selected threshold"
    )
    scan = {
        "id": scan_id,
        "filename": Path(uploaded.name).name,
        "timestamp": timestamp,
        "status": status,
        "model_status": "Trained checkpoint loaded successfully",
        "segmented_pixels": segmented_pixels,
        "total_pixels": total_pixels,
        "percentage": percentage,
        "width": original.width,
        "height": original.height,
        "threshold": threshold,
        "original_png": image_as_png_bytes(original),
        "mask_png": image_as_png_bytes(mask_image),
        "overlay_png": image_as_png_bytes(overlay),
    }
    st.session_state["latest_scan"] = scan
    recent_scans = st.session_state.setdefault("recent_scans", [])
    recent_scans[:] = [item for item in recent_scans if item["id"] != scan_id]
    recent_scans.insert(
        0,
        {
            key: scan[key]
            for key in (
                "id",
                "filename",
                "timestamp",
                "status",
                "segmented_pixels",
                "total_pixels",
                "percentage",
                "width",
                "height",
            )
        },
    )
    del recent_scans[5:]

    st.markdown("#### Synchronized segmentation views")
    image_columns = st.columns(3)
    image_panels = [
        ("Original MRI", original, "Uploaded brain MRI slice"),
        ("Predicted segmentation mask", mask_image, "Binary mask predicted by the U-Net"),
        ("MRI segmentation overlay", overlay, "Red highlights indicate pixels marked by the model"),
    ]
    for column, (title, image, alt) in zip(image_columns, image_panels):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.image(image, alt=alt, width="stretch")
                st.caption(f"{original.width} × {original.height} px")

    st.markdown("#### Quantitative analysis")
    st.badge(status, icon=":material/check_circle:", color="green")
    with st.container(horizontal=True):
        st.metric("Segmented image area (%)", f"{percentage:.2f}%", border=True)
        st.metric("Segmented pixels", f"{segmented_pixels:,}", border=True)
        st.metric("Total image pixels", f"{total_pixels:,}", border=True)
        st.metric("Image dimensions", f"{original.width} × {original.height}", border=True)
    st.caption(
        "Segmented image area (%) = segmented pixels ÷ total image pixels × 100. "
        "This is a measurement of the displayed 2D image, not tumor volume, "
        "cancer probability, or severity."
    )
    st.caption(f"Segmentation threshold: {threshold:.2f}")
    render_classification_panel()

    with st.container(border=True):
        st.markdown("**Research report**")
        st.caption(
            "The PDF includes the source image, predicted mask, overlay, measurements, "
            "model status, clinical-field availability, and limitations."
        )
        render_report_download(scan, key=f"analysis_report_{scan_id[:12]}")

elif page == "Research Reports":
    st.subheader("Research reports")
    latest_scan = st.session_state.get("latest_scan")
    if latest_scan is None:
        st.info(
            "Complete an MRI analysis before generating a report. Reports are generated "
            "from successful inference results in this browser session."
        )
    else:
        with st.container(border=True):
            st.markdown(f"**{latest_scan['filename']}**")
            st.caption(
                f"Analyzed {latest_scan['timestamp']} · "
                f"{latest_scan['width']} × {latest_scan['height']} px"
            )
            st.metric(
                "Segmented image area (%)",
                f"{latest_scan['percentage']:.2f}%",
            )
            st.caption(
                f"{latest_scan['segmented_pixels']:,} segmented pixels of "
                f"{latest_scan['total_pixels']:,} · threshold "
                f"{latest_scan['threshold']:.2f}"
            )
            st.badge(latest_scan["status"], color="green")
            render_report_download(
                latest_scan,
                key=f"research_report_{latest_scan['id'][:12]}",
            )

    st.markdown("#### Recent analysis measurements")
    recent_scans = st.session_state.get("recent_scans", [])
    if not recent_scans:
        st.caption("No successful inference results are available in this session.")
    else:
        for item in recent_scans:
            with st.container(border=True):
                st.markdown(f"**{item['filename']}**")
                st.caption(
                    f"{item['timestamp']} · {item['width']} × {item['height']} px · "
                    f"{item['segmented_pixels']:,} / {item['total_pixels']:,} pixels · "
                    f"{item['percentage']:.2f}% segmented image area"
                )
    st.caption("Analysis history and report images are retained only in this browser session.")

else:
    st.subheader("Model information")
    info_columns = st.columns([1.1, 1])
    with info_columns[0]:
        with st.container(border=True):
            st.markdown("**Model architecture**")
            st.write(
                "The application loads the existing compact U-Net from model.py and "
                "the checkpoint at artifacts/unet_brain_tumor.pth. The model produces "
                "one binary segmentation logit map."
            )
            st.markdown("**Inference preprocessing**")
            st.markdown(
                "- Convert the uploaded slice to RGB.\n"
                "- Resize to 128 × 128 with bilinear interpolation.\n"
                "- Scale pixel values to [0, 1].\n"
                "- Apply sigmoid and the selected mask threshold.\n"
                "- Resize the binary mask to the source dimensions with nearest-neighbor interpolation."
            )
    with info_columns[1]:
        with st.container(border=True):
            st.markdown("**Checkpoint status**")
            if MODEL_PATH.is_file():
                st.success(f"Checkpoint found at `{MODEL_PATH.relative_to(PROJECT_DIR)}`.")
            else:
                st.error(
                    f"Checkpoint missing at `{MODEL_PATH.relative_to(PROJECT_DIR)}`. "
                    "The project will not train or download a model automatically."
                )
            st.markdown("**Clinical scope**")
            st.write(
                "The model is for binary pixel segmentation only. A mask does not "
                "establish whether a finding is a tumor or determine its type, grade, "
                "stage, volume, or severity."
            )
            st.info(
                "No validated classification model or calibrated model-confidence "
                "measure is available."
            )
    st.subheader("Research limitations")
    st.markdown(
        """
        - Only a single 2D image is processed; the system does not analyze 3D volumes.
        - MRI sequence, orientation, acquisition settings, and patient history are not verified.
        - The current model has no validated classification capability or calibrated confidence.
        - Segmented image area (%) is a pixel fraction, not tumor volume, cancer probability,
          or a severity score.
        - Predictions may be inaccurate, particularly for images unlike the training data.
        - This is an educational research prototype, not a medical diagnostic device.
        """
    )
