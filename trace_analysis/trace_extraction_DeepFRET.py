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


def calculate_intensities(movie_path, hdf5_path, inner_radius=2, gap_width=1,
                          roi=(0, 428, 684, 856), outer_radius=5, raw=False):
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


    signal_df = pd.DataFrame(signals,  columns=locs.index)
    # Filter 1: Remove signals with too many negative values
    negative_frac = (signal_df < 0).mean(axis=0)
    signal_df_filtered = signal_df.loc[:, negative_frac < 0.3]

    # Filter 2: Remove constant signals (low standard deviation)
    signal_stds = signal_df_filtered.std(axis=0)
    signal_df_filtered = signal_df_filtered.loc[:, signal_stds >= 10]

    # Save the filtered results
    save_path = movie_path.replace('.tif', '_DeepFRET_intensity.csv')
    signal_df_filtered.to_csv(save_path, index=False)

    # Also filter the background array to match
    valid_columns = signal_df_filtered.columns
    backgrounds_filtered = backgrounds[:, [i for i, col in enumerate(locs.index) if col in valid_columns]]

    return signal_df_filtered, backgrounds_filtered


if __name__ == '__main__':
    calculate_intensities(movie_path="J:/non_competitive/20250319_8nt_NComp_GAP_G_Seal100nM/8nt_NComp_GAP_G_Seal100nM_GAP_G_seal3G_100nM_corrected.tif",
                          hdf5_path="J:/non_competitive/20250319_8nt_NComp_GAP_G_Seal100nM/8nt_NComp_GAP_G_Seal100nM_GAP_G_localization_corrected_boxsize5.hdf5")