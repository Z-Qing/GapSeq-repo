import os
import re
import numpy as np
import pandas as pd
from picasso_utils import one_channel_movie
from picasso.io import save_locs, load_locs
from scipy.spatial import KDTree



def export_locs_PYMEVis(ref_hdf5_path, index, movie_list, pattern, gpu=True, box_size=1.5):
    ref_locs, _ = load_locs(ref_hdf5_path)
    ref_locs = pd.DataFrame(ref_locs)
    ref_coords = ref_locs.loc[index, ['x', 'y']]

    roi = [0, 428, 684, 856]

    locs_list = []
    for movie_path in movie_list:
        nuc = re.search(pattern, os.path.basename(movie_path)).group(1)

        mov = one_channel_movie(movie_path, roi=roi)
        mov.lq_fitting(gpu, min_net_gradient=1000, box=3)
        mov.overlap_prevent(box_radius=box_size)

        locs = pd.DataFrame(mov.locs)
        mov_coords = locs[['x', 'y']]

        # Build KDTree for fast neighbor lookup
        tree = KDTree(mov_coords)

        # Query neighbors within the given radius
        indices = tree.query_ball_point(ref_coords, box_size)

        selected_locs = pd.DataFrame(locs.loc[indices])
        selected_locs['probe'] = nuc

        locs_list.append(selected_locs)


    df = pd.concat(locs_list, axis=0)
    df['x'] = df['x'] - (df['x'].min() - 0.1)
    df['y'] = df['y'] - (df['y'].min() - 0.1)

    df[['x', 'y', 'lpx', 'lpy', 'sx', 'sy']] = df[['x', 'y', 'lpx', 'lpy', 'sx', 'sy']] * 117
    df['sig'] = np.sqrt(np.square(df['sx']) + np.square(df['sy']))

    df.drop(columns=['sx', 'sy'], inplace=True)
    df.rename(columns={'frame': 't', 'photons': 'A', 'lpx': 'error_x', 'lpy': 'error_y'}, inplace=True)

    # df['probe'] = df['probe'].apply(lambda x: x.replace('P', ''))
    # df['probe'] = df['probe'].astype(int)

    nuc_number = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
    df['probe'] = df['probe'].map(nuc_number)

    df.to_csv(os.path.dirname(ref_hdf5_path) + '/{}.csv'.format(index), index=False)

    return



def export_locs_picasso(ref_hdf5_path, index, movie_list, gpu=True, box_size=1.5):
    ref_locs, _ = load_locs(ref_hdf5_path)
    ref_locs = ref_locs.view(np.recarray)
    ref_coords = ref_locs[index]
    ref_coords = np.column_stack((ref_coords.x, ref_coords.y))


    roi = [0, 428, 684, 856]

    for movie_path in movie_list:
        mov = one_channel_movie(movie_path, roi=roi)
        mov.lq_fitting(gpu, min_net_gradient=1000, box=3)
        mov.overlap_prevent(box_radius=box_size)

        mov_coords = np.column_stack((mov.locs['x'], mov.locs['y']))

        # Build KDTree for fast neighbor lookup
        tree = KDTree(mov_coords)

        # Query neighbors within the given radius
        indices = tree.query_ball_point(ref_coords, box_size)


        recarr = mov.locs.view(np.recarray)
        for i in np.arange(len(indices)):
            sliced_recarr = recarr[indices[i]]
            folder_path = os.path.dirname(movie_path) + '/index_{}'.format(index[i])

            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            base_name = os.path.basename(movie_path)
            save_locs(folder_path + '/' + base_name.replace('.tif', '.hdf5'), sliced_recarr, mov.info)

    return




dir_path = 'H:/non_competitive/20250320_8nt_NComp_GAP_A_Seal100nM'
hdf5_path = "H:/non_competitive/20250320_8nt_NComp_GAP_A_Seal100nM\8nt_NComp_GAP_A_Seal100nM_GAP_A_localization-1_corrected_boxsize5.hdf5"

movie_list = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.tif')]
movie_list = [x for x in movie_list if 'localization' not in x and 'Localization' not in x]

export_locs_picasso(ref_hdf5_path=hdf5_path, index=[175, 450, 606, 744, 852, 1096], movie_list=movie_list, box_size=1.5)
