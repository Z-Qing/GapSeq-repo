import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy.interpolate import UnivariateSpline

def subtract_background_deepFRET(arr, deg=2, s=1e6, by="row"):
    if arr.ndim == 2:
        # Handle 2D case (single image)
        bg = _compute_background(arr, deg, s, by)
    elif arr.ndim == 3:
        with Pool() as pool:
            bg = np.stack(pool.map(
                partial(_compute_background, deg=deg, s=s, by=by),
                arr
            ))
    else:
        raise ValueError("Input must be 2D (image) or 3D (movie).")

    filtered = arr - bg
    filtered = np.clip(filtered, 0, 65535)

    return filtered.astype(np.uint16)


def _compute_background(frame, deg, s, by):
    """Helper function to compute background for a single frame."""
    if by == "column":
        # Column-wise spline
        ix = np.arange(frame.shape[0])
        bg = np.column_stack([
            UnivariateSpline(ix, frame[:, i], k=deg, s=s)(ix)
            for i in range(frame.shape[1])
        ])
    elif by == "row":
        # Row-wise spline
        ix = np.arange(frame.shape[1])
        bg = np.row_stack([
            UnivariateSpline(ix, frame[i, :], k=deg, s=s)(ix)
            for i in range(frame.shape[0])
        ])
    else:
        raise ValueError(f"Invalid method: {by}. Use 'row', 'column', or 'filter'.")

    return bg


if __name__ == '__main__':
    from tifffile import imread, imwrite
    image = imread("G:/background_remove/GAP_G_Ncomp_seal100nM_seal3A_100nM-1_red.tif")
    filtered = subtract_background_deepFRET(image, s=1e6, deg=1)
    imwrite("G:/background_remove/GAP_G_Ncomp_seal100nM_seal3A_100nM-1_red_filtered_deg1.tif", filtered)