# coregistration_svg.py
# Requires: nibabel, numpy, scipy, pillow
# Example usage:
#   create_coregistration_svg("T1.nii.gz", "DWI.nii.gz", "coreg.svg")

import io
import base64
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from PIL import Image, ImageDraw, ImageFont
import sys


def _normalize_to_uint8(img, pmin=1, pmax=99):
    img = np.asarray(img, dtype=float)
    if img.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(img[np.isfinite(img)], (pmin, pmax))
    if hi <= lo:
        hi = img.max() if img.max() > lo else lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _orient_axial_slice(slice_xy):
    """
    Make image rows = Y (anterior-posterior), cols = X (left-right),
    with anterior up. Works as a generic transpose+flip for axial/sagittal/coronal
    when slices are provided as (X,Y), (Y,Z), or (X,Z) respectively.
    """
    img = slice_xy.T
    img = np.flipud(img)
    return img


def _make_montage_pil(image_list, bg=(0, 0, 0)):
    # Horizontally concatenate PIL images, aligning top and padding background
    widths = [im.width for im in image_list]
    heights = [im.height for im in image_list]
    total_w = sum(widths)
    max_h = max(heights) if heights else 0
    out = Image.new("RGB", (total_w, max_h), color=bg)
    x = 0
    for im in image_list:
        out.paste(im, (x, 0))
        x += im.width
    return out


def _stack_rows_pil(row_images, bg=(0, 0, 0)):
    widths = [im.width for im in row_images]
    heights = [im.height for im in row_images]
    max_w = max(widths) if widths else 0
    total_h = sum(heights)
    out = Image.new("RGB", (max_w, total_h), color=bg)
    y = 0
    for im in row_images:
        # left-align rows
        out.paste(im, (0, y))
        y += im.height
    return out


def _draw_text_with_outline(draw, xy, text, font, fill=(255, 255, 255), outline=(0, 0, 0)):
    x, y = xy
    # 1-pixel outline
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def _draw_label_on_pil(im, text, pad=6, font=None):
    draw = ImageDraw.Draw(im)
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    text_w = draw.textlength(text, font=font)
    try:
        text_h = font.getsize(text)[1] if font is not None else 18
    except Exception:
        text_h = 18
    x = im.width - text_w - pad
    y = im.height - text_h - pad
    _draw_text_with_outline(draw, (x, y), text, font)
    return im


def _draw_number_on_pil(im, text, pad=6, font=None):
    draw = ImageDraw.Draw(im)
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    x = pad
    y = pad
    _draw_text_with_outline(draw, (x, y), text, font)
    return im


def create_coregistration_svg(t1_nii_path,
                              dwi_nii_path,
                              out_svg_path,
                              slices=None,
                              n_slices=7,
                              fps=1.5):
    """
    Create an animated SVG showing three rows of montages (horizontal rows of slices):
      - top row: axial (Z)  slices
      - middle row: sagittal (X) slices
      - bottom row: coronal (Y)  slices
    Each row shows n_slices (default 7) across the middle 60% of that axis.
    Slice numbers (axis=index) are drawn on each slice.
    The animation switches between the T1w montage and the DWI montage.
    """
    t1_img = nib.load(t1_nii_path)
    dwi_img = nib.load(dwi_nii_path)

    t1_data = t1_img.get_fdata(dtype=np.float32)
    dwi_data = dwi_img.get_fdata(dtype=np.float32)

    if dwi_data.ndim == 4:
        dwi_mean = np.nanmean(dwi_data, axis=3)
    else:
        dwi_mean = dwi_data

    t1_aff = t1_img.affine
    dwi_aff = dwi_img.affine
    vox2vox = np.linalg.inv(dwi_aff) @ t1_aff
    matrix = vox2vox[:3, :3]
    offset = vox2vox[:3, 3]

    t1_shape = t1_data.shape
    dwi_resampled = affine_transform(
        dwi_mean,
        matrix=matrix,
        offset=offset,
        output_shape=t1_shape,
        order=1,
        mode='nearest'
    )

    # Determine indices per axis (Z axial, X sagittal, Y coronal)
    x_dim, y_dim, z_dim = t1_shape[0], t1_shape[1], t1_shape[2]

    def _middle_indices(dim, n):
        start = int(dim * 0.2)
        stop = int(dim * 0.8)
        if stop <= start:
            start = 0
            stop = dim - 1
        return np.linspace(start, stop, n, dtype=int)

    if slices is None:
        z_indices = _middle_indices(z_dim, n_slices)
        x_indices = _middle_indices(x_dim, n_slices)
        y_indices = _middle_indices(y_dim, n_slices)
    else:
        # if user provided a single list, treat as Z for axial; for simplicity require None to change axes
        arr = np.asarray(slices, dtype=int)
        z_indices = arr
        x_indices = arr
        y_indices = arr

    # prepare font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

    # Build three rows of PIL images for T1 and DWI
    def build_rows_for_volume(vol_data, label_prefix=""):
        # For each axis produce list of PIL images (slices with numbers)
        axial_imgs = []
        sagittal_imgs = []
        coronal_imgs = []

        # axial (Z): slice = vol[:, :, z] shape (X, Y)
        for z in z_indices:
            slice_xy = vol_data[:, :, int(z)]
            oriented = _orient_axial_slice(slice_xy)
            u8 = _normalize_to_uint8(oriented)
            rgb = np.stack([u8, u8, u8], axis=-1)
            im = Image.fromarray(rgb)
            _draw_number_on_pil(im, f"Z={int(z)}", font=font)
            axial_imgs.append(im)

        # sagittal (X): slice = vol[x, :, :] shape (Y, Z)
        for x in x_indices:
            slice_yz = vol_data[int(x), :, :]
            oriented = _orient_axial_slice(slice_yz)
            u8 = _normalize_to_uint8(oriented)
            rgb = np.stack([u8, u8, u8], axis=-1)
            im = Image.fromarray(rgb)
            _draw_number_on_pil(im, f"X={int(x)}", font=font)
            sagittal_imgs.append(im)

        # coronal (Y): slice = vol[:, y, :] shape (X, Z)
        for y in y_indices:
            slice_xz = vol_data[:, int(y), :]
            oriented = _orient_axial_slice(slice_xz)
            u8 = _normalize_to_uint8(oriented)
            rgb = np.stack([u8, u8, u8], axis=-1)
            im = Image.fromarray(rgb)
            _draw_number_on_pil(im, f"Y={int(y)}", font=font)
            coronal_imgs.append(im)

        # make horizontal montages
        row1 = _make_montage_pil(axial_imgs)
        row2 = _make_montage_pil(sagittal_imgs)
        row3 = _make_montage_pil(coronal_imgs)

        # optionally add small row labels at left (Axial/Sagittal/Coronal)
        _draw_text_with_outline(ImageDraw.Draw(row1), (6, row1.height - 22), "Axial", font, fill=(255,255,255))
        _draw_text_with_outline(ImageDraw.Draw(row2), (6, row2.height - 22), "Sagittal", font, fill=(255,255,255))
        _draw_text_with_outline(ImageDraw.Draw(row3), (6, row3.height - 22), "Coronal", font, fill=(255,255,255))

        # stack vertically
        full = _stack_rows_pil([row1, row2, row3])
        # modality label bottom-right
        _draw_label_on_pil(full, label_prefix, font=font)
        return full

    t1_full = build_rows_for_volume(t1_data, label_prefix="T1w")
    dwi_full = build_rows_for_volume(dwi_resampled, label_prefix="DWI reference")

    frames = [t1_full, dwi_full]
    png_b64_list = []
    for im in frames:
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        png_b64_list.append(b64)

    height, width = frames[0].height, frames[0].width
    n_frames = len(frames)
    frame_dur = 2.0 / fps
    total_dur = n_frames * frame_dur
    key_times = ";".join([f"{i/n_frames:.6f}" for i in range(n_frames + 1)])

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" ')
    svg_parts.append(f'     xmlns:xlink="http://www.w3.org/1999/xlink">')
    svg_parts.append(f'<desc>Animated coregistration montages (rows: axial, sagittal, coronal).')
    svg_parts.append(f' Slices Z: {list(z_indices)}, X: {list(x_indices)}, Y: {list(y_indices)}</desc>')

    for i, b64 in enumerate(png_b64_list):
        svg_parts.append(f'<image x="0" y="0" width="{width}" height="{height}" xlink:href="data:image/png;base64,{b64}" opacity="0">')
        values = ";".join([("1" if j == i else "0") for j in range(n_frames)] + [("1" if 0 == i else "0")])
        svg_parts.append(f'  <animate attributeName="opacity" dur="{total_dur}s" values="{values}" keyTimes="{key_times}" repeatCount="indefinite" />')
        svg_parts.append('</image>')

    svg_parts.append('</svg>')

    svg_text = "\n".join(svg_parts)

    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_text)

    return out_svg_path

def create_html_report(svg_path, out_html_path):
    """
    Create a simple HTML report embedding the SVG file.
    """
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html lang="en">')
    html_parts.append('<head><meta charset="UTF-8"><title>Coregistration Report</title></head>')
    html_parts.append('<body>')
    html_parts.append('<h1>Coregistration Report</h1>')
    html_parts.append(svg_content)
    html_parts.append('</body>')
    html_parts.append('</html>')

    html_text = "\n".join(html_parts)

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    return out_html_path

# If run as script, simple CLI example
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python coregistration_svg.py T1.nii.gz DWI.nii.gz out.svg")
    else:
        create_coregistration_svg(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"SVG saved to {sys.argv[3]}")
        create_html_report(sys.argv[3], sys.argv[3].replace('.svg', '.html'))
        print(f"HTML report saved to {sys.argv[3].replace('.svg', '.html')}")