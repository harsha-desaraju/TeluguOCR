"""Skew estimation by Hough transform, and the rotation that corrects it.

This code is a copy of the code from deskew python package (with little modifications).
"""

import cv2
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

if TYPE_CHECKING:
    from typing import TypeAlias

    ImageType: TypeAlias = npt.NDArray[np.integer[Any] | np.floating[Any]]
    ImageTypeUint64: TypeAlias = npt.NDArray[np.uint8]
    ImageTypeFloat64: TypeAlias = npt.NDArray[np.float64]
else:
    ImageType = np.ndarray
    ImageTypeUint64 = np.ndarray
    ImageTypeFloat64 = np.ndarray


def determine_skew_dev(
    image: ImageType,
    sigma: float = 3.0,
    num_peaks: int = 20,
    min_angle: float | None = None,  # -np.pi / 2,
    max_angle: float | None = None,  # np.pi / 2,
    min_deviation: float = np.pi / 180,
    angle_pm_90: bool = False,
) -> tuple[
    np.float64 | None,
    tuple[
        tuple[ImageTypeUint64, list[list[np.float64]], ImageTypeFloat64],
        tuple[list[Any], list[np.float64], list[np.float64]],
        tuple[dict[np.float64, int], dict[np.float64, int]],
    ],
]:
    """Calculate skew angle."""
    num_angles = round(np.pi / min_deviation)
    imagergb = rgba2rgb(image) if len(image.shape) == 3 and image.shape[2] == 4 else image  # type: ignore[no-untyped-call]
    img = rgb2gray(imagergb) if len(imagergb.shape) == 3 else imagergb
    edges = canny(img, sigma=sigma)  # type: ignore[no-untyped-call]
    out, angles, distances = hough_line(edges, np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False))  # type: ignore[no-untyped-call]
    hough_line_out = (out, angles, distances)

    hspace, angles_peaks, dists = hough_line_peaks(  # type: ignore[no-untyped-call]
        out,
        angles,
        distances,
        num_peaks=num_peaks,
        threshold=0.05 * np.max(out),
    )
    hough_line_peaks_out = (hspace, angles_peaks, dists)

    if len(angles_peaks) == 0:
        return None, (hough_line_out, hough_line_peaks_out, ({}, {}))

    freqs_original: dict[np.float64, int] = {}
    for peak in angles_peaks:
        freqs_original.setdefault(peak, 0)
        freqs_original[peak] += 1

    angles_peaks_corrected = [
        (a % np.pi - np.pi / 2) if angle_pm_90 else ((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        for a in angles_peaks
    ]
    angles_peaks_filtred = (
        [a for a in angles_peaks_corrected if a >= min_angle]
        if min_angle is not None
        else angles_peaks_corrected
    )
    angles_peaks_filtred = (
        [a for a in angles_peaks_filtred if a <= max_angle] if max_angle is not None else angles_peaks_filtred
    )
    if not angles_peaks_filtred:
        return None, (hough_line_out, hough_line_peaks_out, ({}, {}))

    freqs: dict[np.float64, int] = {}
    for peak in angles_peaks_filtred:
        freqs.setdefault(peak, 0)
        freqs[peak] += 1

    angle = max(freqs, key=freqs.get)  # type: ignore[arg-type]

    return (
        angle,
        (hough_line_out, hough_line_peaks_out, (freqs_original, freqs)),
    )


def determine_skew(
    image: ImageType,
    sigma: float = 3.0,
    num_peaks: int = 20,
    num_angles: int | None = None,
    angle_pm_90: bool = False,
    min_angle: float | None = None,
    max_angle: float | None = None,
    min_deviation: float = 1.0,
    max_skew: float | None = 10.0,
) -> np.float64 | None:
    """
    Calculate skew angle.

    Parameters
    ----------
    image: np.ndarray
        Input image
    sigma: float
        Standard deviation of Gaussian filter
    num_peaks: int
        Number of peaks to detect
    num_angles: int
        Number of angles to consider
    angle_pm_90: bool
        Consider angles in the range [-180, 180] instead of [-90, 90]
    min_angle: float
        Minimum angle to consider
    max_angle: float
        Maximum angle to consider
    min_deviation: float
        Minimum deviation between angles
    max_skew: float
        Reject (return None) beyond this many degrees. Scans are not skewed by 30
        degrees; such an estimate means the Hough peak came from a page edge or a
        binarisation artifact, and acting on it destroys the page. None disables.

    Returns
    -------
    float
        Skew angle in degrees, None if no skew will be found
    """
    if num_angles is not None:
        min_deviation = 180 / num_angles
        warnings.warn("num_angles is deprecated, please use min_deviation", DeprecationWarning, stacklevel=2)

    angle, _ = determine_skew_dev(
        image,
        sigma=sigma,
        num_peaks=num_peaks,
        min_angle=np.deg2rad(min_angle) if min_angle is not None else None,
        max_angle=np.deg2rad(max_angle) if max_angle is not None else None,
        min_deviation=np.deg2rad(min_deviation),
        angle_pm_90=angle_pm_90,
    )
    if angle is None:
        return None
    degrees = np.rad2deg(angle)
    if max_skew is not None and abs(degrees) > max_skew:
        return None
    return degrees




def rotate_image(image: np.ndarray, angle: float | None) -> np.ndarray:
    """Rotate about the centre onto a canvas grown to fit, padding with white."""
    if not angle:
        return image

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    # ceil, not int: truncating leaves the canvas a fraction short and clips a corner.
    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))

    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


if __name__ == '__main__':

    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    path_to_image = "/Users/xai/Desktop/page1.png"

    image = Image.open(path_to_image)

    angle = determine_skew(np.array(image))
    rotated_image = rotate_image(np.array(image), angle)

    plt.imshow(rotated_image)
    plt.show()