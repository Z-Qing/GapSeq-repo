import numpy as np
from tifffile import imread, imwrite
import multiprocessing
import pandas as pd
from skimage.morphology import white_tophat, footprint_rectangle



def top_hat_filter(image, i, size=5):
    res = white_tophat(image, footprint_rectangle((size, size)))
    return res, i


def trace_extraction(ref_locs, mov_path, box_size=3, roi=(0, 428, 684, 856), save=True):
    ref_locs = pd.read_hdf(ref_locs, 'locs')

    movie = imread(mov_path)
    movie = movie[:, roi[0]:roi[2], roi[1]:roi[3]]
    param_list = [(movie[i, :, :], i) for i in range(movie.shape[0])]

    filtered = np.zeros_like(movie)
    with multiprocessing.Pool() as pool:
        for res, i in pool.starmap(top_hat_filter,  param_list):
            filtered[i, :, :] = res

    if save:
        imwrite(mov_path.replace('.tif', '_filtered.tif'), filtered)

    traces = pd.DataFrame(np.zeros((movie.shape[0], len(ref_locs))), columns=ref_locs.index)
    for i in ref_locs.index:
        pos = np.round(ref_locs.loc[i][['x', 'y']]).astype(int)

        x_start = pos['x'] - box_size//2
        x_end = pos['x'] + box_size//2
        y_start = pos['y'] - box_size//2
        y_end = pos['y'] + box_size//2

        one_trace = filtered[:, y_start:y_end, x_start:x_end].sum(axis=(1, 2))
        traces[i] = one_trace


    return traces


if __name__ == '__main__':
    trace = trace_extraction(
        ref_locs="G:/background_remove/CAP_lib_1nM_seq_CAP_localization_corrected_picasso_bboxes.hdf5",
        mov_path="G:/background_remove/CAP_lib_1nM_seq_CAP_binding-1_corrected.tif")

    trace.to_csv("G:/background_remove/CAP_lib_1nM_seq_CAP_binding-1_corrected_traces.csv", index=False)