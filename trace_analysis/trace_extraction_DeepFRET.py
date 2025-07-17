import numpy as np
from tifffile import imread
import pandas as pd

def create_circular_masks(image_shape, yx, inner_radius, gap_width, outer_radius):
    """
    adapted from DeepFRET
    """
    yy, xx = yx
    yi, xi = np.indices(image_shape)
    dist_sq = (yy - yi) ** 2 + (xx - xi) ** 2

    # Create masks
    center_mask = dist_sq <= inner_radius ** 2
    inner_plus_gap = inner_radius + gap_width
    bg_outer_mask = dist_sq <= outer_radius ** 2
    bg_inner_mask = dist_sq <= inner_plus_gap ** 2

    # Background is the ring between outer radius and inner+gap radius
    bg_mask = np.logical_and(bg_outer_mask, ~bg_inner_mask)

    return center_mask, bg_mask


def calculate_intensities(movie_path, hdf5_path, inner_radius=3, gap_width=1,
                          roi=(0, 428, 684, 856), outer_radius=6, raw=False):
    """
    adapted from DeepFRET
    raw:
        Whether to return raw signal/background get_intensities. Otherwise will
        return signal-background and background as zeroes.
    """
    locs = pd.read_hdf(hdf5_path, key='locs')
    yx_coords = locs[['y', 'x']].to_numpy()

    image_stack = imread(movie_path)
    image_stack = image_stack[:, roi[0]:roi[2], roi[1]:roi[3]]

    n_frames = image_stack.shape[0]
    n_localizations = len(yx_coords)

    signals = np.zeros((n_frames, n_localizations))
    backgrounds = np.zeros((n_frames, n_localizations))

    for i, (y, x) in enumerate(yx_coords):
        # Create masks for this localization
        center_mask, bg_mask = create_circular_masks(
            image_shape=image_stack.shape[1:],
            yx=(y, x),
            inner_radius=inner_radius,
            gap_width=gap_width,
            outer_radius=outer_radius
        )

        # Calculate intensities
        roi_pixels = image_stack[:, center_mask]
        roi_sum = np.sum(roi_pixels, axis=1)

        bg_values = image_stack[:, bg_mask]
        bg_median = np.median(bg_values, axis=1)
        n_roi_pixels = np.sum(center_mask)
        bg_sum = bg_median * n_roi_pixels

        if raw:
            signals[:, i] = roi_sum
            backgrounds[:, i] = bg_sum
        else:
            signals[:, i] = roi_sum - bg_sum
            backgrounds[:, i] = 0


    save_path = hdf5_path.replace('.hdf5', '_DeepFRET_intensity.csv')
    signal_df = pd.DataFrame(signals,  columns=locs.index)
    # filter out signal that are negative in general (come from false
    # localizations at the edge because of padding when do alignment)
    negative_frac = (signal_df < 0).mean(axis=0)
    signal_df_filtered = signal_df.loc[:, negative_frac < 0.5]
    signal_df_filtered.to_csv(save_path, index=False)

    return signals, backgrounds


if __name__ == '__main__':
    calculate_intensities(movie_path="G:/CAP binding/20250707_CAP_library_2.5nM/CAP_library_2.5nM_CAP_binding_2.5nM_halfdiluted-2_corrected.tif",
                          hdf5_path="G:/CAP binding/20250707_CAP_library_2.5nM/CAP_library_2.5nM_library_localization_corrected_first_frame_locs.hdf5")