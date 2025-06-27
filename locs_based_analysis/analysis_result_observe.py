import os
import re
import numpy as np
import pandas as pd
from picasso_utils import one_channel_movie
from picasso.io import save_locs, load_locs
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings


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


co_localization_rate(ref_path="G:/Cap_library_24062025/20250624_CAP_1base_seqN/co_local_rate/CAP_1base_seqN_libary_localization_corrected_green_first_frame_locs.hdf5",
                   anneal_path="G:/Cap_library_24062025/20250624_CAP_1base_seqN/co_local_rate/CAP_1base_seqN_CAP_reanneal_corrected_green_first_frame_locs.hdf5")


from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler

def first_index_with_all_following_below(values, threshold):
    b = values < threshold
    start = 0
    end = len(values) - 1
    result = len(values) - 1

    while start < end:
        middle = (start + end) // 2
        if np.all(b[middle:]):
            result = middle
            end = middle - 1
        else:
            start = middle + 1

    if result == len(values) - 1:
        raise ValueError("gradient threshold is too low")

    return result


def find_x_interaction(values):
    zero_crossings = np.where(values == 0)[0].tolist()
    # Check for sign changes
    sign_changes = np.where(values[:-1] * values[1:] < 0)[0].tolist()

    zero_crossings.extend(sign_changes)
    zero_crossings = sorted(list(set(zero_crossings)))

    if len(zero_crossings) <= 1:
        raise ValueError("zero_crossings can't be found")

    return zero_crossings[1]


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


def base_calling(path, correct_pick=None, gradient_threshold=0.05, bin_width=10,
                 maximum_length=1000, exp_type='competitive', display=False):
    param = pd.read_csv(path, index_col=0)
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

    transition_point_idx = first_index_with_all_following_below(np.abs(second_deriv), gradient_threshold)
    #transition_point_idx = find_x_interaction(second_deriv)
    transition_point = position[transition_point_idx]
    print(transition_point)


    # ----------------- confidence VS accuracy rate plot -------------------
    if exp_type == 'competitive':
        choice, diff = competitive_selection(param, transition_point)
    elif exp_type == 'non_competitive':
        choice, diff = non_competitive_selection(param, transition_point)
    else:
        raise ValueError

    if display and isinstance(correct_pick, str):
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].plot(position, density, label='scaled density')
        ax[0, 1].plot(position, second_deriv,label='second derivative')
        ax[0, 1].vlines(transition_point, min(second_deriv), max(second_deriv), colors='green', linestyles='dashed')

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
                    rate = summary.loc[correct_pick] / summary.sum()
                    accuracy_rate.append(rate)
                else:
                    accuracy_rate.append(0)

                thresholds.append(t)
                molecule_number.append(len(selected_choice))

        ax[1, 0].plot(thresholds, accuracy_rate, '-o', label='accuracy rate')
        ax[1, 1].plot(thresholds, molecule_number, '-o', label='molecular number')
        plt.legend(loc='best')
        #plt.savefig('E:/Thesis/chapter4_GapSeq/figures/threshold determination/threshold_confidence.png', dpi=600)
        plt.show()




    return pd.DataFrame({'choice': choice, 'diff': diff})



# base_calling("G:/time_vs_accoracy/csv_files/8ntGAP_T_Ncomp_seal100nM_Localization_corrected_picasso_bboxes_neighbour_counting_radius2_200.csv",
#              exp_type='non_competitive',  correct_pick='A', display=True)


def time_VS_accuracy(dir_path, correct_pick, confidence, exp_type='non_competitive'):
    files = [x for x in os.listdir(dir_path) if x.endswith('.csv')]

    frame_num = []
    accuracy_rate = []
    molecule_num = []
    for f in files:
        num = f.split('.')[0]
        num = num.split('_')[-1]
        if num.isdigit():
            result = base_calling(os.path.join(dir_path, f), exp_type=exp_type)
            frame_num.append(int(num))

            result = result[result['diff'] > confidence]
            summary = result['choice'].value_counts()

            number = summary.sum()
            molecule_num.append(number)
            accuracy_rate.append(summary.loc[correct_pick] / number)
        else:
            warnings.warn("detected other types of csv file")
            continue

    fig, ax = plt.subplots(2, 1, )
    ax[0].plot(frame_num, accuracy_rate, 'o')
    ax[0].title.set_text('accuracy rate')
    ax[0].set_xlabel('frame')
    ax[0].set_ylabel('accuracy')

    ax[1].plot(frame_num, molecule_num, 'o')
    ax[1].title.set_text('molecular number')
    ax[1].set_xlabel('frame')
    ax[1].set_ylabel('molecular number')

    fig.tight_layout()
    plt.show()

    df = pd.DataFrame({'frame': frame_num, 'accuracy': accuracy_rate, 'molecule_number': molecule_num})
    df.sort_values('frame', inplace=True)
    df.to_csv(os.path.join(dir_path, 'frame_vs_accuracy_rate.csv'))
    return


#time_VS_accuracy("G:/time_vs_accoracy/csv_files", correct_pick='A', confidence=0.4)
