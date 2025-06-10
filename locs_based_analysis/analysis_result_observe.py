import os
import re
import numpy as np
import pandas as pd
from picasso_utils import one_channel_movie
from picasso.io import save_locs, load_locs
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def export_locs_picasso(ref_hdf5_path, index, movie_list, gpu=True, box_size=2):
    ref_locs, _ = load_locs(ref_hdf5_path)
    ref_locs = ref_locs.view(np.recarray)
    ref_coords = ref_locs[index]
    ref_coords = np.column_stack((ref_coords.x, ref_coords.y))

    roi = [0, 428, 684, 856]

    for movie_path in movie_list:
        if movie_path.endswith('.tif'):
            mov = one_channel_movie(movie_path, roi=roi)
            mov.lq_fitting(gpu, min_net_gradient=1000, box=5)
            mov_coords = np.column_stack((mov.locs['x'], mov.locs['y']))

        elif movie_path.endswith('.hdf5'):
            mov_locs, _ = load_locs(movie_path)
            mov_coords = np.column_stack((mov_locs['x'], mov_locs['y']))
            
        else:
            raise NotImplementedError

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


def co_localization_rate(ref_path, anneal_path, box_size=2):
    ref_locs, _ = load_locs(ref_path)
    anneal_locs, _ = load_locs(anneal_path)

    ref_locs = pd.DataFrame(ref_locs)
    anneal_locs = pd.DataFrame(anneal_locs)

    print(len(ref_locs))
    print(len(anneal_locs))

    ref_coords = ref_locs[['x', 'y']].to_numpy()
    anneal_coords = anneal_locs[['x', 'y']].to_numpy()

    tree = KDTree(anneal_coords)
    # Query neighbors within the given radius
    indices = tree.query_ball_point(ref_coords, box_size)

    neighbor_counts = [len(neigh) for neigh in indices]
    neighbour_exist = [x for x in neighbor_counts if x > 0]

    co_local_rate = len(neighbour_exist) / len(ref_coords)
    print(co_local_rate)

    return


co_localization_rate(ref_path="G:/co-localization_rate/20250608_CAPbinding_Af647_1nM/CAPbinding_Af647_1nM_CAP_localization-1_locs.hdf5",
                  anneal_path="G:/co-localization_rate/20250608_CAPbinding_Af647_1nM/CAPbinding_Af647_1nM_CAP_degen200nM_reannealed-1_locs.hdf5")


from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler

def first_index_with_all_following_below(nums, threshold):
    candidate_idx = -1
    max_after = -float('inf')  # Tracks the maximum number after current index

    for i in range(len(nums) - 1, -1, -1):  # Iterate backward
        num = nums[i]
        if num >= threshold:
            max_after = num
        elif max_after < threshold:
            candidate_idx = i  # Update candidate index if everything after is < threshold

    # Edge case: If all numbers after index 0 are < threshold, return 0
    if candidate_idx == -1 and nums and nums[0] >= threshold:
        return 0
    return candidate_idx if candidate_idx != -1 else None



def competitive_selection(param, threshold):
    b = np.sum(param.to_numpy() > threshold, axis=1)
    b = b >= 3
    param = param.loc[b]

    choice = param.idxmin(axis=1)

    sorted = param.to_numpy()
    sorted.sort(axis=1)
    diff = sorted[:, 1] - sorted[:, 0]
    diff = diff / diff.max()

    return choice, diff


def non_competitive_selection(param, threshold):
    b = np.sum(param.to_numpy() > threshold, axis=1)
    b = b >= 1
    param = param.loc[b]

    choice = param.idxmax(axis=1)

    sorted = param.to_numpy()
    sorted.sort(axis=1)
    diff = sorted[:, -1] - sorted[:, -2] # the difference between largest number and second largest
    diff = diff / diff.max()

    return choice, diff


def base_calling(path, correct_pick, gradient_threshold=0.05, bin_width=10,
                 maximum_length=1000, competitive=False):
    param = pd.read_csv(path, index_col=0, header=[0, 1])
    param = param['regular']
    #remove fiducial markers
    param = param.loc[param.max(axis=1) < maximum_length]

    # ----------------- threshold selection ------------------------------
    locs_counts = param.to_numpy().flatten()
    position = np.arange(0, max(locs_counts) + bin_width, bin_width)
    # plt.hist(locs_counts, bins=position)
    # plt.show()

    kernel = gaussian_kde(locs_counts)
    density = kernel(position)
    density = MinMaxScaler((0, 1000)).fit_transform(density.reshape(-1, 1)).flatten()

    first_deriv = np.gradient(density, position)
    second_deriv = np.gradient(first_deriv, position)

    # plt.plot(position, density)
    # plt.show()
    # plt.plot(position, first_deriv)
    # plt.show()
    # plt.plot(position, second_deriv)
    # plt.show()

    transition_point_idx = first_index_with_all_following_below(np.abs(second_deriv), gradient_threshold)
    transition_point = position[transition_point_idx]
    print(transition_point)


    # ----------------- confidence VS accuracy rate plot -------------------
    if competitive == True:
        choice, diff = competitive_selection(param, transition_point)
    else:
        choice, diff = non_competitive_selection(param, transition_point)

    thresholds = []
    accuracy_rate = []
    molecule_number = []
    for t in np.arange(0, 1, 0.1):
        selected_choice = choice.loc[diff > t]
        if len(selected_choice) == 0:
            break
        else:
            summary = selected_choice.value_counts()
            if correct_pick in summary.index:
                rate = summary.loc[correct_pick] / np.sum(summary)
                accuracy_rate.append(rate)
            else:
                accuracy_rate.append(0)

            thresholds.append(t)
            molecule_number.append(len(selected_choice))

    plt.plot(thresholds, accuracy_rate, '-o')
    plt.show()

    plt.plot(thresholds, molecule_number, '-o')
    plt.show()


    return


# base_calling("G:/8ntGAP_T_Ncomp_seal100nM_Localization_corrected_picasso_bboxes_neighbour_counting_radius2_linked.csv",
#             correct_pick='A')

# base_calling("G:/8nt_comp_GAP_G_GAP_G_localization_corrected_boxsize5_neighbour_counting_radius2_linked_paper.csv",
#              competitive=True, correct_pick='C')