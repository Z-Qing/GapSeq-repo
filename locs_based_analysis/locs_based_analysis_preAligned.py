import numpy as np
import pandas as pd
from picasso.io import save_locs, load_locs
from picasso.lib import ensure_sanity
import re
import os
from scipy.spatial import KDTree
from picasso_utils import one_channel_movie
from sklearn.cluster import DBSCAN



def neighbour_counting(ref_points, mov_points, nuc, box_radius=1.5):
    # Extract x, y coordinates
    ref_coords = np.column_stack((ref_points['x'], ref_points['y']))
    mov_coords = np.column_stack((mov_points['x'], mov_points['y']))

    # Build KDTree for fast neighbor lookup
    tree = KDTree(mov_coords)

    # Query neighbors within the given radius
    indices = tree.query_ball_point(ref_coords, box_radius)

    params = pd.DataFrame(index=np.arange(ref_points.shape[0]))
    params[(nuc, 'event_num')] = [mov_points.iloc[ind]['event_num'].sum() for ind in indices]
    params[(nuc, 'count')] = [mov_points.iloc[ind]['count'].sum() for ind in indices]
    
    params.columns = pd.MultiIndex.from_tuples(params.columns)
    params.index.name = 'ref_index'

    return params



def locs_based_analysis_preAligned(ref_path, mov_list, pattern, search_radius=1.5, mov_gradient=1000,
                                   gpu=True, ref_roi=None, ref_gradient=400,
                                   roi=None, save_hdf5=False):
    if ref_path.endswith('.hdf5'):
        ref_locs, _ = load_locs(ref_path)

    elif ref_path.endswith('.tif'):
        ref = one_channel_movie(ref_path, roi=ref_roi, frame_range=0)
        ref.lq_fitting(GPU=gpu, min_net_gradient=ref_gradient, box=5)
        ref.overlap_prevent(box_radius=search_radius * 2)

        ref_locs = ensure_sanity(ref.locs, ref.info) # make sure the csv and hdf5 match
        save_locs(ref_path.replace('.tif', '.hdf5'), ref_locs, ref.info)

    else:
        raise ValueError('un-supported format')

    nuc_cluster_param = {}
    for movie_path in mov_list:
        mov = one_channel_movie(movie_path, roi=roi)
        mov.movie_format(baseline=20)
        mov.lq_fitting(gpu, min_net_gradient=mov_gradient, box=5)
        mov.dbscan(eps=0.5, min_samples=20)
        mov.trace_analysis(display=False)

        nuc = re.search(pattern, os.path.basename(movie_path)).group(1)
        nuc_cluster_param[nuc] = mov.cluster_param

        if save_hdf5:
            save_locs(movie_path.replace('.tif', '.hdf5'), locs, mov.info)

    # ----------------------- neighbour counting ----------------------
    total_params = []
    for nuc in nuc_cluster_param.keys():
        param = neighbour_counting(ref_locs, nuc_cluster_param[nuc], nuc, box_radius=search_radius)
        total_params.append(param)

    total_params = pd.concat(total_params, axis=1)

    return total_params


def process_analysis_Localization(dir_path, pattern, ref_path=None):
    files = [x for x in os.listdir(dir_path) if x.endswith('.tif')]

    if ref_path == None:
        ref = [x for x in files if 'Localization' in x or 'localization' in x]
        if len(ref) != 1:
            raise ValueError("There should be one and only one reference file in the directory")
        else:
            files.remove(ref[0])
            ref_keyword = ref[0].replace('.tif', '')
        ref = os.path.join(dir_path, ref[0])
    else:
        ref = ref_path
        ref_keyword = os.path.basename(ref_path).replace('.tif', '')


    mov_list = [os.path.join(dir_path, x) for x in files if ('Localization' not in x) and ('localization' not in x)]

    counts = locs_based_analysis_preAligned(ref, mov_list, pattern=pattern, search_radius=1.5, gpu=True,
                                            roi=[0, 428, 684, 856], ref_roi=[0, 0, 684, 428],
                                            ref_gradient=400, mov_gradient=1000, save_hdf5=False)

    counts.to_csv(dir_path + '/{}_neighbour_counting.csv'.format(ref_keyword))


    return


def process_analysis_ALEX(dir_path):
    files = os.listdir(dir_path)
    ref = [x for x in files if x.endswith('.hdf5')]
    if len(ref) != 1:
        raise ValueError("There should be one and only one reference file in the directory")
    else:
        ref = os.path.join(dir_path, ref[0])
        ref_keyword = ref[0].replace('.hdf5', '')

    subfolder_pattern = {}
    subfolder_pattern['4D5X'] = r'_S4D5([A-Z])300nM'
    subfolder_pattern['4X5D'] = r'_S4([A-Z])5D300nM'

    for folder, pattern in subfolder_pattern.items():
        mov_list = os.listdir(os.path.join(dir_path, folder))
        mov_list = [os.path.join(dir_path, folder, x) for x in mov_list if x.endswith('.tif')]

        counts = locs_based_analysis_preAligned(ref, mov_list=mov_list, pattern=pattern, search_radius=1.5, gpu=True,
                                       roi=[0, 428, 684, 856], save_hdf5=False, mov_gradient=1500)

        counts.to_csv(os.path.join(dir_path, folder) + '{}_/neighbour_counting.csv'.format(ref_keyword))

    return



if __name__ == "__main__":
    #process_analysis_ALEX("H:/jagadish_data/20250308_IPE_trans_NTP200Exp15")
    process_analysis_Localization('H:/photobleaching/20250322_8nt_NComp_photobleaching2/median_25',
                                  ref_path="H:/photobleaching/20250322_8nt_NComp_photobleaching2/median_25/"
                                           "8nt_NComp_photobleaching2_Gap_G_localization_corrected.tif",
                                  pattern=r'_seal3([A-Z])_100nM')
