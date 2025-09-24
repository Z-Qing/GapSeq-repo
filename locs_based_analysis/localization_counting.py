import numpy as np
import pandas as pd
from picasso.io import save_locs, load_locs
import re
import os
from scipy.spatial import KDTree
from picasso_utils import one_channel_movie
import warnings
import matplotlib.pyplot as plt


def is_well_spread_by_bundles(
    frames: np.ndarray,
    *,
    min_bundles: int = 5,
    max_longest_run_frac: float = 0.5,
    total_frames_ref: int | None = None,
    min_run_len_frames: int = 1,
    min_total_locs: int = 100
) -> bool:
    """
    Return True if the cluster passes (keep), False if it fails (reject).

    frames            : 1D array of frame indices for the cluster's localizations.
    min_bundles       : require at least this many consecutive-frame bundles.
    max_longest_run_frac : longest bundle length must be <= this fraction of total_frames_ref.
    total_frames_ref  : reference number of frames (e.g., total movie frames).
                        If None, uses the cluster's covered span (max(frames)-min(frames)+1).
    min_run_len_frames: only count runs of at least this many frames as a bundle.
    min_total_locs    : if total localizations < this, auto-keep (or tweak as you like).
    """
    f = np.asarray(frames, dtype=int)
    if f.size < min_total_locs:
        return True

    uf = np.unique(f)
    if uf.size == 0:
        return False
    if uf.size == 1:
        # Single-frame cluster is a single bundle of length 1
        longest = 1
        num_bundles = 1 if 1 >= min_run_len_frames else 0
    else:
        # Find consecutive runs over unique frames
        breaks = np.diff(uf) != 1
        starts = np.r_[0, np.nonzero(breaks)[0] + 1]
        ends   = np.r_[starts[1:] - 1, uf.size - 1]

        run_lens = uf[ends] - uf[starts] + 1  # length in frames
        # Only count "real" bundles
        real = run_lens >= min_run_len_frames
        run_lens = run_lens[real]
        num_bundles = int(run_lens.size)
        longest = int(run_lens.max()) if num_bundles > 0 else 0

    # Reference frame count: movie length (recommended) or cluster span
    if total_frames_ref is None:
        total_frames_ref = int(uf[-1] - uf[0] + 1)

    # Rule A: enough bundles
    rule_a = num_bundles >= min_bundles
    # Rule B: longest bundle small enough
    rule_b = (longest / max(total_frames_ref, 1)) <= max_longest_run_frac

    return bool(rule_a and rule_b)


def neighbour_counting(ref_points, mov_points, nuc, frame_number,
                       search_radius=2):
    # Extract x, y coordinates
    ref_coords = np.column_stack((ref_points['x'], ref_points['y']))
    mov_points = pd.DataFrame.from_records(mov_points)
    mov_coords = np.column_stack((mov_points['x'], mov_points['y']))

    # Build KDTree for fast neighbor lookup
    tree = KDTree(mov_coords)

    # Query neighbors within the given radius
    indices = tree.query_ball_point(ref_coords, search_radius)

    # get the frames for the neighbour localisations
    neighbour_counts = []
    keep_list = []
    for neigh in indices:
        neighbour_counts.append(len(neigh))

        neigh_frames = mov_points.loc[neigh, 'frame']
        if len(neigh_frames) > 0:
            well_spread= is_well_spread_by_bundles(neigh_frames, total_frames_ref=frame_number)
            keep_list.append(well_spread)

            # if not well_spread:
            #     y = np.zeros(frame_number)
            #     y[neigh_frames] = 1.0
            #     plt.plot(np.arange(frame_number), y)
            #     plt.show()

        else:
            keep_list.append(True)

    counts = pd.DataFrame({nuc: neighbour_counts})
    counts.index.name = 'ref_index'

    keep = pd.DataFrame({nuc: keep_list})
    keep.index.name = 'ref_index'

    return counts, keep




def locs_based_analysis_preAligned(ref_path, mov_list, pattern, search_radius=2,
                                   mov_gradient=1000, max_frame=np.inf, mov_baseline=400,
                                   gpu=True, ref_roi=None, ref_gradient=400, roi=None, save_hdf5=False):
    if ref_path.endswith('.hdf5'):
        ref_locs, _ = load_locs(ref_path)
        ref_locs = pd.DataFrame.from_records(ref_locs)
        if 'group' in ref_locs.columns:
            ref_locs = ref_locs.groupby(by='group').mean()

    elif ref_path.endswith('.tif'):
        ref = one_channel_movie(ref_path, roi=ref_roi, frame_range=0)
        ref.movie_format()
        ref.lq_fitting(GPU=gpu, gradient=ref_gradient, box=5)
        ref.overlap_prevent(box_radius=search_radius * 2)

        ref_locs = ref.locs
        save_locs(ref_path.replace('.tif', '.hdf5'), ref_locs, ref.info)

    elif ref_path.endswith('.csv'):
        ref_locs = pd.read_csv(ref_path)

    else:
        raise ValueError('please provide the address of .hdf5 or .tif file')

    nuc_locs = {}
    for movie_path in mov_list:
        f = os.path.basename(movie_path)
        try:
            nuc = re.search(pattern, f).group(1)
        except:
            warnings.warn('cannot find nucleotide for {}'.format(f))
            continue

        if movie_path.endswith('.hdf5'):
            locs, info = load_locs(movie_path)
            locs = locs[locs.frame < max_frame]
            nuc_locs[nuc] = locs

        elif movie_path.endswith('.tif'):
            mov = one_channel_movie(movie_path, roi=roi, frame_range=(0, max_frame))
            mov.movie_format(baseline=mov_baseline)
            mov.lq_fitting(gpu, gradient=mov_gradient, box=5)
            nuc_locs[nuc] = mov.locs

            if save_hdf5:
                save_locs(movie_path.replace('.tif', '.hdf5'), mov.locs, mov.info)

        else:
            raise NotImplementedError

    # ----------------------- neighbour counting ----------------------
    total_params = []
    total_keep = []
    for nuc in nuc_locs.keys():
        frame_number = max(nuc_locs[nuc]['frame']) + 1
        param, keep = neighbour_counting(ref_locs, nuc_locs[nuc], nuc, frame_number, search_radius=search_radius)
        total_params.append(param)
        total_keep.append(keep)

    # remove molecules that have stuck seals
    counting_params = pd.concat(total_params, axis=1)  # counts per nuc
    keep_params_df = pd.concat(total_keep, axis=1)  # booleans per nuc (aligned by ref_index)
    keep_mask = keep_params_df.all(axis=1, skipna=False)  # <-- pandas .all keeps the index

    print("Num rows dropped:", (~keep_mask).sum())
    counting_params = counting_params.loc[keep_mask].copy()


    return counting_params



def process_analysis_Localization(dir_path, pattern, ref_path=None, target_format='.tif',
                                  localization_keyword='localization', max_frame=np.inf,
                                   search_radius=2.0, gradient=1000, gpu=True, save_hdf5=False):
    files = [x for x in os.listdir(dir_path) if x.endswith(target_format)]

    if ref_path is None:
        ref = [x for x in files if localization_keyword in x]
        if len(ref) != 1:
            raise ValueError("There should be one and only one reference file in the directory")
        else:
            files.remove(ref[0])
            ref_keyword = ref[0].replace(target_format, '')
        ref = os.path.join(dir_path, ref[0])
        mov_list = [os.path.join(dir_path, x) for x in files if localization_keyword not in x]

    else:
        ref = ref_path
        ref_keyword = os.path.basename(ref_path).split('.')[0]
        mov_list = [os.path.join(dir_path, x) for x in files if localization_keyword.lower() not in x.lower()]

    counts = locs_based_analysis_preAligned(ref, mov_list, pattern=pattern, search_radius=search_radius, gpu=gpu,
                                            roi=[0, 428, 684, 856], ref_roi=[0, 0, 684, 428], max_frame=max_frame,
                                            ref_gradient=400, mov_gradient=gradient, save_hdf5=save_hdf5)
    counts.to_csv(dir_path + '/{}_neighbour_counting_radius{}_{}.csv'.format(ref_keyword, search_radius, max_frame))

    return



def process_analysis_ALEX(dir_path, ref_path=None, search_radius=2.0, mov_gradient=1000, gpu=True,
                          max_frame=np.inf, save_hdf5=False):
    files = os.listdir(dir_path)
    if ref_path is None:
        ref = [x for x in files if x.endswith('.hdf5')]
        if len(ref) != 1:
            raise ValueError("There should be one and only one reference file in the directory")
        else:
            ref_keyword = ref[0].replace('.hdf5', '')
            ref = os.path.join(dir_path, ref[0])
    else:
        ref = ref_path


    subfolder_pattern = {}
    subfolder_pattern['4D5X'] = r'_S4D5([A-Z])_'
    subfolder_pattern['4X5D'] = r'_S4([A-Z])5D_'

    for folder, pattern in subfolder_pattern.items():
        mov_list = os.listdir(os.path.join(dir_path, folder))
        mov_list = [os.path.join(dir_path, folder, x) for x in mov_list if x.endswith('.tif')]

        counts = locs_based_analysis_preAligned(ref, mov_list=mov_list, pattern=pattern,
                                    search_radius=search_radius, gpu=gpu, ref_gradient=400,
                                    max_frame=max_frame, ref_roi=[0, 0, 684, 428], mov_gradient=mov_gradient,
                                       roi=[0, 428, 684, 856], save_hdf5=save_hdf5)

        counts.to_csv(os.path.join(dir_path, folder) + '/{}_neighbour_counting_radius{}_linked.csv'.format(ref_keyword, search_radius))

    return




if __name__ == "__main__":
    #process_analysis_ALEX("G:/20250405_IPE_NTP200_ALEX_exp29", gradient=750, gpu=True)

    process_analysis_Localization("G:/new_accuracy_table/5base/pos8",
                                  ref_path=None,
                                  localization_keyword='localization', # use for find reference molecules or exclude the localization movie
                                  gpu=True,
                                  pattern= r'_seal5([A-Z])_',
                                  #r'_S6A_(\d+(?:\.\d+)?(?:nM|uM))_corrected',, #r'_([a-zA-Z]+)500nM_',
                                  #r'_(\d+(?:\.\d+)?(?:nM|uM))_corrected',
                                  max_frame=np.inf,
                                  save_hdf5=True,
                                  target_format='.hdf5',
                                  search_radius=2.0, gradient=1000)
