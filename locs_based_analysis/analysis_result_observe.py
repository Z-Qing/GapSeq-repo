import os
import re
import numpy as np
import pandas as pd
from picasso_utils import one_channel_movie
from picasso.io import save_locs, load_locs
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def export_locs_PYMEVis(ref_hdf5_path, index, movie_list, pattern, gpu=True, box_size=1.5):
    ref_locs, _ = load_locs(ref_hdf5_path)
    ref_locs = pd.DataFrame(ref_locs)
    ref_coords = ref_locs.loc[index, ['x', 'y']]

    roi = [0, 428, 684, 856]

    locs_list = []
    for movie_path in movie_list:
        nuc = re.search(pattern, os.path.basename(movie_path)).group(1)

        mov = one_channel_movie(movie_path, roi=roi)
        mov.lq_fitting(gpu, min_net_gradient=1000, box=5)
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



def export_locs_picasso(ref_hdf5_path, index, movie_list, gpu=True, box_size=2):
    ref_locs, _ = load_locs(ref_hdf5_path)
    ref_locs = ref_locs.view(np.recarray)
    ref_coords = ref_locs[index]
    ref_coords = np.column_stack((ref_coords.x, ref_coords.y))


    roi = [0, 428, 684, 856]

    for movie_path in movie_list:
        mov = one_channel_movie(movie_path, roi=roi)
        mov.lq_fitting(gpu, min_net_gradient=1000, box=5)

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




# dir_path = 'H:\competitive/20250325_8nt_comp_GAP_G'
# hdf5_path = "H:\competitive/20250325_8nt_comp_GAP_G\8nt_comp_GAP_G_GAP_G_localization_corrected.hdf5"
#
# movie_list = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.tif')]
# movie_list = [x for x in movie_list if 'localization' not in x and 'Localization' not in x]
# #
# export_locs_picasso(ref_hdf5_path=hdf5_path, index=[115, 135, 151, 472, 599, 749, 829, 920,
#                                                     931, 979, 1015, 1022, 1045,
#                                                     1263, 1264, 1378, 1490, 1541, 1690, 1871, 473], movie_list=movie_list, box_size=2)


def nucleotide_selection_non_competitive(path, correct_pick=None, maximum_high_counts=1000,
                                         minimum_diff=200):
    counts = pd.read_csv(path, index_col=0)

    counts = counts.loc[np.any(counts, axis=1)]

    b1 = counts.max(axis=1) < maximum_high_counts
    filtered_counts = counts.loc[b1.values, :].copy()

    second_largest = filtered_counts.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)

    diff = filtered_counts.max(axis=1) - second_largest
    b2 = diff > minimum_diff
    # filtered_counts['diff'] = diff

    filtered_counts = filtered_counts.loc[b2.values, :]

    result = filtered_counts.idxmax(axis=1)

    if correct_pick is None:
        print(result)
        return

    else:
        total_num = result.sum()
        correct_pick_num = result.loc[correct_pick]
        accuracy_rate = correct_pick_num / total_num

        return accuracy_rate, total_num


# nucleotide_selection_non_competitive("H:/non_competitive/20250325_8nt_Noncomp_GAP_T/corrected_movies/"
#                                      "8nt_Noncomp_GAP_T_GAP_T_localization_corrected_neighbour_counting.csv")



def nucleotide_selection_competitive(path, correct_pick, maximum_high_counts=1000,
                                     threshold=None, save_csv=True):
    complementary_nuc = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    counts = pd.read_csv(path, index_col=0)

    result = counts.copy()
    result['pick'] = 'None'
    result['minimum'] = counts.idxmin(axis=1).replace(complementary_nuc)

    second_minimum_counts = counts.apply(lambda row: row.nlargest(3).iloc[-1], axis=1)
    result['diff'] = second_minimum_counts - counts.min(axis=1)

    # filtering based on noise to signal ratio
    # result['noise_to_signal'] = result['diff'] / (second_minimum_counts + 0.000001)
    # filtered_counts = counts.loc[result['noise_to_signal'] > 0.6, :]

    b0 = np.sum(counts > 30, axis=1) >= 3
    filtered_counts = counts.loc[b0.values, :]

    b1 = ~(np.sum(filtered_counts > maximum_high_counts, axis=1) >= 1)
    filtered_counts = filtered_counts.loc[b1.values, :]

    diff = result.loc[filtered_counts.index, 'diff']

    if threshold is not None:
        b = diff > threshold
        final_counts = filtered_counts.loc[b.values, :]
        final_selection = final_counts.idxmin(axis=1)
        print(final_selection.loc[final_selection != correct_pick])


    rate_list = []
    total_num_list = []
    for min_count_diff in np.arange(10, 200, 10):
        b2 = diff > min_count_diff

        temp_counts = filtered_counts.loc[b2.values, :]
        selected_nuc = temp_counts.idxmin(axis=1)
        result.loc[temp_counts.index, 'pick'] = selected_nuc.replace(complementary_nuc)

        result_summary = selected_nuc.value_counts()

        correct_pick_num = result_summary.loc[correct_pick]
        total_num = result_summary.sum()

        total_num_list.append(result_summary.sum())
        rate_list.append(correct_pick_num / total_num)

    plt.plot(np.arange(10, 200, 10), rate_list, '-o')
    plt.show()

    plt.plot(np.arange(10, 200, 10), total_num_list, '-o')
    plt.show()

    df = pd.DataFrame({'diff threshold': np.arange(10, 200, 10),
                       'accuracy rate': np.round(rate_list, 6), 'number of molecules': total_num_list})
    if save_csv:
        df.to_csv(os.path.dirname(path) + '/diff_noise_to_signal.csv')

    return df



# nucleotide_selection_competitive("H:\competitive/20250325_8nt_comp_GAP_C\
# 8nt_comp_GAP_C_GAP_C_localization_corrected_neighbour_counting_radius2.csv",
#                                  correct_pick='G', save_csv=True)
