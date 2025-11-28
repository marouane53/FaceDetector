# FaceDetector

Detect and blur faces in photos with modern Python tooling. The script scans an input
folder, crops detected faces to a separate directory, writes blurred copies to an
output folder, and preserves EXIF metadata where possible.

## Requirements
- Python 3.9+
- OpenCV with Haar cascade data (bundled with `opencv-python`)

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run the blurring workflow from the project root:

```bash
python blur.py \
  --input-dir photo/original \
  --output-dir photo/blur \
  --faces-dir photo/faces
```

### Useful options
- `--recursive` — include images in subdirectories.
- `--blur-kernel` — odd integer kernel size for Gaussian blur (default: 23).
- `--min-size` — minimum face size in pixels for detection (default: 30).
- `--frontal-cascade` / `--profile-cascade` — override Haar cascade XML paths.
- `--log-level` — set verbosity (DEBUG, INFO, WARNING, ERROR).

Images without EXIF data will be processed normally but metadata copying will be
skipped with an informational log entry.

## Project layout
- `blur.py` — CLI entry point and face-blurring workflow.
- `requirements.txt` — Python dependencies.
- `photo/original` — expected default input directory (create as needed).
- `photo/blur` — blurred image output directory.
- `photo/faces` — extracted face crops.
