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




# dir_path = 'H:/competitive/20250325_8nt_comp_GAP_C'
# hdf5_path = "H:/competitive/20250325_8nt_comp_GAP_C/8nt_comp_GAP_C_GAP_C_localization_corrected_.hdf5"
#
# movie_list = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.tif')]
# movie_list = [x for x in movie_list if 'localization' not in x and 'Localization' not in x]
#
# export_locs_picasso(ref_hdf5_path=hdf5_path, index=[389, 1173, 1360, 1940, 1644, 121], movie_list=movie_list, box_size=1.5)



def counts_filtering(path, maximum_high_counts=900, min_event_diff=10,
                        min_count_diff=50):
    params = pd.read_csv(path, index_col=0, header=[0, 1])

    counts = params.loc[:, (slice(None), 'count')]
    counts.columns = counts.columns.droplevel(1)

    event_num = params.loc[:, (slice(None), 'event_num')]
    event_num.columns = event_num.columns.droplevel(1)

    # second_largest_event_num = event_num.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    # plt.hist(event_num.max(axis=1) - second_largest_event_num, bins=50)
    # plt.show()
    #
    # second_largest_locs_num = counts.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    # plt.hist(counts.max(axis=1) - second_largest_locs_num, bins=50)
    # plt.show()


    #filter out fiduical markers
    b1 = np.sum(counts > maximum_high_counts, axis=1) < 2
    filtered_counts = counts.loc[b1.values, :]
    filtered_events = event_num.loc[b1.values, :]



    # maximum counts and event should be the same
    selected_nuc_counts = filtered_counts.idxmax(axis=1)
    selected_nuc_events = filtered_events.idxmax(axis=1)
    b3 = selected_nuc_counts == selected_nuc_events

    filtered_counts = filtered_counts.loc[b3.values, :]
    filtered_events = filtered_events.loc[b3.values, :]

    # reach minimum diff in locs counts
    second_largest_counts = filtered_counts.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    diff = filtered_counts.max(axis=1) - second_largest_counts
    b2 = diff > min_count_diff

    filtered_counts = filtered_counts.loc[b2.values, :]
    filtered_events = filtered_events.loc[b2.values, :]

    # reach minimum diff in event number
    second_largest_event_num = filtered_events.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    diff = filtered_events.max(axis=1) - second_largest_event_num
    b4 = diff > min_event_diff

    filtered_counts = filtered_counts.loc[b4.values, :]
    filtered_events = filtered_events.loc[b4.values, :]

    selected_nuc = filtered_counts.idxmax(axis=1)
    print(selected_nuc.value_counts())
    # print(filtered_counts[selected_nuc != 'G'])
    # print(filtered_events[selected_nuc != 'G'])

    return



counts_filtering_V2("H:/photobleaching/20250322_8nt_NComp_photobleaching2/median_25/"
                 "8nt_NComp_photobleaching2_Gap_G_localization_corrected_neighbour_counting_radius1.5.csv")