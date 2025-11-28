from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import piexif


FaceBox = Tuple[int, int, int, int]


@dataclass
class CascadePaths:
    """Container for cascade classifier paths."""

    frontal: Path
    profile: Path

    @classmethod
    def default(cls) -> "CascadePaths":
        haar_root = Path(cv2.data.haarcascades)
        return cls(
            frontal=haar_root / "haarcascade_frontalface_alt.xml",
            profile=haar_root / "haarcascade_profileface.xml",
        )


@dataclass
class BlurConfig:
    input_dir: Path
    output_dir: Path
    faces_dir: Path
    cascades: CascadePaths
    blur_kernel: Tuple[int, int]
    min_size: Tuple[int, int]
    recursive: bool


class FaceBlurrer:
    def __init__(self, config: BlurConfig) -> None:
        self.config = config
        self.frontal_cascade = self._load_cascade(config.cascades.frontal)
        self.profile_cascade = self._load_cascade(config.cascades.profile)

    def run(self) -> None:
        image_paths = self._gather_images(
            self.config.input_dir, recursive=self.config.recursive
        )
        if not image_paths:
            logging.warning("No images found in %s", self.config.input_dir)
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.faces_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            logging.info("Processing %s", image_path)
            self._process_image(image_path)

    def _process_image(self, image_path: Path) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning("Skipping unreadable image %s", image_path)
            return

        result_image = image.copy()

        frontal_faces = self._detect_faces(
            result_image, self.frontal_cascade, min_size=self.config.min_size
        )
        self._save_faces(image, frontal_faces, image_path.stem)
        result_image = self._blur_faces(result_image, frontal_faces)

        profile_faces = self._detect_faces(
            result_image, self.profile_cascade, min_size=self.config.min_size
        )
        self._save_faces(result_image, profile_faces, image_path.stem, prefix="profile_")
        result_image = self._blur_faces(result_image, profile_faces)

        output_path = self.config.output_dir / image_path.name
        cv2.imwrite(str(output_path), result_image)
        self._copy_exif(image_path, output_path)

    def _detect_faces(
        self, image: np.ndarray, cascade: cv2.CascadeClassifier, min_size: Tuple[int, int]
    ) -> List[FaceBox]:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.equalizeHist(gray_img)

        faces = cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=min_size,
        )
        return [tuple(map(int, face)) for face in faces]

    def _blur_faces(self, image: np.ndarray, faces: Sequence[FaceBox]) -> np.ndarray:
        result = image.copy()
        for (x, y, w, h) in faces:
            sub_face = result[y : y + h, x : x + w]
            blurred = cv2.GaussianBlur(sub_face, self.config.blur_kernel, sigmaX=30)
            result[y : y + h, x : x + w] = blurred
        return result

    def _save_faces(
        self, image: np.ndarray, faces: Iterable[FaceBox], stem: str, prefix: str = ""
    ) -> None:
        for idx, (x, y, w, h) in enumerate(faces):
            face_img = image[y : y + h, x : x + w]
            face_filename = f"{prefix}{stem}_face{idx+1}.jpg"
            face_path = self.config.faces_dir / face_filename
            cv2.imwrite(str(face_path), face_img)

    def _copy_exif(self, source: Path, destination: Path) -> None:
        try:
            exif_dict = piexif.load(str(source))
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(destination))
        except piexif.InvalidImageDataError:
            logging.info("No EXIF data to copy for %s", source)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Unable to copy EXIF from %s: %s", source, exc)

    def _gather_images(self, directory: Path, recursive: bool) -> List[Path]:
        patterns = ["*.jpg", "*.jpeg", "*.png"]
        paths: List[Path] = []
        iterator = directory.rglob if recursive else directory.glob
        for pattern in patterns:
            paths.extend(sorted(iterator(pattern)))
        return paths

    @staticmethod
    def _load_cascade(path: Path) -> cv2.CascadeClassifier:
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise FileNotFoundError(f"Unable to load cascade from {path}")
        return cascade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Detect and blur faces in images while preserving metadata."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("photo/original"),
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("photo/blur"),
        help="Directory where blurred images will be saved.",
    )
    parser.add_argument(
        "--faces-dir",
        type=Path,
        default=Path("photo/faces"),
        help="Directory where cropped faces will be saved.",
    )
    parser.add_argument(
        "--frontal-cascade",
        type=Path,
        default=CascadePaths.default().frontal,
        help="Path to the frontal face Haar cascade XML file.",
    )
    parser.add_argument(
        "--profile-cascade",
        type=Path,
        default=CascadePaths.default().profile,
        help="Path to the profile face Haar cascade XML file.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=23,
        help="Kernel size (odd integer) for Gaussian blur.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=30,
        help="Minimum face size (pixels) for detection.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images in subdirectories recursively.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for console output.",
    )
    return parser.parse_args()


def normalize_kernel(kernel: int) -> Tuple[int, int]:
    if kernel % 2 == 0:
        kernel += 1
    return kernel, kernel


def normalize_min_size(min_size: int) -> Tuple[int, int]:
    return max(1, min_size), max(1, min_size)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s"
    )

    config = BlurConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        faces_dir=args.faces_dir,
        cascades=CascadePaths(frontal=args.frontal_cascade, profile=args.profile_cascade),
        blur_kernel=normalize_kernel(args.blur_kernel),
        min_size=normalize_min_size(args.min_size),
        recursive=args.recursive,
    )

    FaceBlurrer(config).run()


if __name__ == "__main__":
    main()
