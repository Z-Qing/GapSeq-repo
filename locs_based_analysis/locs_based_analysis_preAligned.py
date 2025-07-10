import numpy as np
import pandas as pd
from numpy.ma.core import max_filler
from picasso.io import save_locs, load_locs
import re
import os
from scipy.spatial import KDTree
from picasso_utils import one_channel_movie
from sklearn.cluster import DBSCAN
from picasso.postprocess import link
import warnings

def neighbour_counting(ref_points, mov_points, nuc, search_radius=2):
    # Extract x, y coordinates
    ref_coords = np.column_stack((ref_points['x'], ref_points['y']))
    mov_coords = np.column_stack((mov_points['x'], mov_points['y']))

    # Build KDTree for fast neighbor lookup
    tree = KDTree(mov_coords)

    # Query neighbors within the given radius
    indices = tree.query_ball_point(ref_coords, box_radius)

    # Count neighbors for each reference point
    neighbor_counts = [len(neigh) for neigh in indices]


    # Convert to DataFrame
    params = pd.DataFrame({nuc: neighbor_counts})
    params.index.name = 'ref_index'

    return params



def locs_based_analysis_preAligned(ref_path, mov_list, pattern, search_radius=2,
                                   mov_gradient=1000, max_frame=np.inf, mov_baseline=400,
                                   gpu=True, ref_roi=None, ref_gradient=400, roi=None, save_hdf5=False):
    if ref_path.endswith('.hdf5'):
        ref_locs, _ = load_locs(ref_path)

    elif ref_path.endswith('.tif'):
        ref = one_channel_movie(ref_path, roi=ref_roi, frame_range=0)
        ref.lq_fitting(GPU=gpu, gradient=ref_gradient, box=5)
        ref.overlap_prevent(box_radius=search_radius * 2)

        ref_locs = ref.locs
        save_locs(ref_path.replace('.tif', '.hdf5'), ref_locs, ref.info)

    else:
        raise ValueError('please provide the address of .hdf5 or .tif file')

    nuc_locs = {}
    #nuc_info = {}
    for movie_path in mov_list:
        try:
            nuc = re.search(pattern, os.path.basename(movie_path)).group(1)
        except:
            warnings.warn('cannot find nucleotide for {}'.format(movie_path))
            continue

        if movie_path.endswith('.hdf5'):
            locs, info = load_locs(movie_path)
            locs = locs[locs.frame < max_frame]
            nuc_locs[nuc] = locs
            #nuc_info[nuc] = info

        elif movie_path.endswith('.tif'):
            mov = one_channel_movie(movie_path, roi=roi, frame_range=max_frame)
            mov.movie_format(baseline=mov_baseline)
            mov.lq_fitting(gpu, gradient=mov_gradient, box=5)

            #nuc_info[nuc] = mov.info
            nuc_locs[nuc] = mov.locs

            if save_hdf5:
                save_locs(movie_path.replace('.tif', '.hdf5'), mov.locs, mov.info)

        else:
            raise NotImplementedError

    # ----------------------- neighbour counting ----------------------
    total_params = []
    for nuc in nuc_locs.keys():
        param = neighbour_counting(ref_locs, nuc_locs[nuc], nuc, search_radius=search_radius)
        total_params.append(param)

    counting_params = pd.concat(total_params, axis=1)

    return counting_params



def process_analysis_Localization(dir_path, pattern, ref_path=None, target_format='.tif',
                                  localization_keyword='localization', max_frame=np.inf,
                                   search_radius=2, gradient=1000, gpu=True, save_hdf5=False):
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



def process_analysis_ALEX(dir_path, search_radius=2, gradient=1000, gpu=True):
    files = os.listdir(dir_path)
    ref = [x for x in files if x.endswith('.hdf5')]
    if len(ref) != 1:
        raise ValueError("There should be one and only one reference file in the directory")
    else:
        ref_keyword = ref[0].replace('.hdf5', '')
        ref = os.path.join(dir_path, ref[0])


    subfolder_pattern = {}
    subfolder_pattern['4D5X'] = r'_S4D5([A-Z])_'
    subfolder_pattern['4X5D'] = r'_S4([A-Z])5D_'

    for folder, pattern in subfolder_pattern.items():
        mov_list = os.listdir(os.path.join(dir_path, folder))
        mov_list = [os.path.join(dir_path, folder, x) for x in mov_list if x.endswith('.tif')]

        counts = locs_based_analysis_preAligned(ref, mov_list=mov_list, pattern=pattern,
                                    search_radius=search_radius, gpu=gpu, ref_gradient=400,
                                       roi=[0, 428, 684, 856], save_hdf5=False, mov_gradient=gradient)

        counts.to_csv(os.path.join(dir_path, folder) + '/{}_neighbour_counting_radius{}_linked.csv'.format(ref_keyword, search_radius))

    return



if __name__ == "__main__":
    #process_analysis_ALEX("G:/20250405_IPE_NTP200_ALEX_exp29", gradient=750, gpu=True)
    process_analysis_Localization("G:/20250625_Steve_SpcBio_1G2ASpc_LC_counting",
                                  ref_path="G:/20250625_Steve_SpcBio_1G2ASpc_LC_counting/Steve_SpcBio_1G2ASpc_Localisation_corrected_picasso_bboxes_overlappingBoxesRemoved.hdf5",
                                  localization_keyword='localization', # use for find reference molecules or exclude the localization movie
                                  gpu=True,
                                  pattern=r'_seal(.*?)_', # r'degen100nM_([A-Za-z])_',
                                  max_frame=np.inf,
                                  save_hdf5=False,
                                  target_format='.tif',
                                  search_radius=2, gradient=1000)
