import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit


def polynomial(x, *coeffs):
    return np.polyval(coeffs[::-1], x)  # coeffs ordered from highest to lowest degree

def threshold_selection(data, degree=12, bin_size=10, display=True):
    # Create histogram
    counts, bin_edges = np.histogram(data, bins=np.arange(0, data.max() + bin_size, bin_size))
    positions = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Calculate first derivative
    first_deriv = np.gradient(counts, positions)

    # Fit polynomial to first derivative
    p0 = [1] * (degree + 1)
    popt, _ = curve_fit(polynomial, positions, first_deriv, p0=p0)
    poly_coeffs = popt[::-1]  # Convert to standard np.poly order
    first_deriv_poly = np.poly1d(poly_coeffs)

    # Find global minimum (using fitted polynomial)
    x_fine = np.linspace(positions.min(), positions.max(), 1000)
    y_fine = first_deriv_poly(x_fine)
    global_min_idx = np.argmin(y_fine)
    global_min_x = x_fine[global_min_idx]

    # Candidate 1: First zero crossing after minimum
    # We'll scan the polynomial values after the minimum
    post_min_x = x_fine[global_min_idx:]
    post_min_y = y_fine[global_min_idx:]

    zero_crossings = np.where(np.diff(np.sign(post_min_y)) > 0)[0]
    candidate1 = post_min_x[zero_crossings[0]] if len(zero_crossings) > 0 else None

    # Candidate 2: First local maximum after minimum
    # Find where derivative changes from positive to negative
    is_max = (post_min_y[1:-1] > post_min_y[:-2]) & (post_min_y[1:-1] > post_min_y[2:])
    maxima = post_min_x[1:-1][is_max]
    candidate2 = maxima[0] if len(maxima) > 0 else None

    # Select the earliest occurring candidate
    transition_x = None
    if candidate1 and candidate2:
        transition_x = min(candidate1, candidate2)
    elif candidate1:
        transition_x = candidate1
    elif candidate2:
        transition_x = candidate2
    else:
        transition_x = global_min_x  # Fallback

    if display:
        fig, ax = plt.subplots(2, 1)
        ax[1].plot(positions, first_deriv, 'b-', alpha=0.5, label='First Derivative (data)')
        ax[1].plot(x_fine, y_fine, 'r-', label='Polynomial Fit')
        ax[1].axhline(0, color='gray', linestyle='--')
        ax[1].axvline(global_min_x, color='green', linestyle=':', label='Global Min')
        if candidate1:
            ax[1].axvline(candidate1, color='purple', linestyle=':', label='Zero Crossing')
        if candidate2:
            ax[1].axvline(candidate2, color='orange', linestyle=':', label='First Max')
        ax[1].axvline(transition_x, color='black', linewidth=2, label='Selected Transition')
        ax[0].bar(positions, counts, width=bin_size)
        ax[0].axvline(transition_x, color='black', linewidth=2)
        plt.legend()
        plt.show()

    return transition_x


def competitive_selection(param, threshold):
    b = np.sum(param.to_numpy() > threshold, axis=1)
    b = b >= 3
    param = param.loc[b]

    choice = param.idxmin(axis=1)

    sorted = param.to_numpy()
    sorted.sort(axis=1)
    diff = sorted[:, 1] - sorted[:, 0]
    diff = np.where(diff==0, 1, diff)
    # avoid value 0 (in this scenario, the diff is 0
    # or 1 won't make much difference)
    diff = np.log(diff)

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
    diff = np.where(diff == 0, 1, diff)
    diff = np.log(diff)

    diff = diff / diff.max()

    return choice, diff


def base_calling(path, maximum_length, exp_type, correct_pick=None, threshold=None,
                 bin_width=5, display=False, save_results=False):
    param = pd.read_csv(path, index_col=0)
    #remove fiducial markers
    param = param.loc[~(param.min(axis=1) > maximum_length)]

    # ----------------- threshold selection ------------------------------
    if type(threshold) == int or type(threshold) == float:
        transition_point = threshold
    else:
        locs_counts = param.to_numpy().flatten()
        transition_point = threshold_selection(locs_counts, bin_size=bin_width)
        print(transition_point)

    # ----------------- confidence VS accuracy rate plot -------------------
    if exp_type == 'competitive':
        choice, confidence = competitive_selection(param, transition_point)
    elif exp_type == 'non-competitive':
        choice, confidence = non_competitive_selection(param, transition_point)
    else:
        raise ValueError


    if save_results:
        results = param.loc[choice.index].copy()
        results['calling'] = choice
        results['confidence'] = confidence
        results.to_csv(path.replace('.csv', '_base_calling_result.csv'), index=True)


    if display and isinstance(correct_pick, str):
        fig, ax = plt.subplots(2, 1)

        thresholds = []
        accuracy_rate = []
        molecule_number = []
        for t in np.arange(0, 1, 0.1):
            selected_choice = choice.loc[confidence > t]
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

        ax[0].plot(thresholds, accuracy_rate, '-o', label='accuracy rate')
        ax[1].plot(thresholds, molecule_number, '-o', label='molecular number')
        plt.legend(loc='best')
        plt.show()

        df = pd.DataFrame({'threshold': thresholds, 'accuracy_rate': accuracy_rate,
                           'molecule_number': molecule_number})
        print(df)

    return pd.DataFrame({'choice': choice, 'confidence': confidence})



def time_VS_accuracy(dir_path, correct_pick, minimum_confidence, exp_type, display=True):
    files = [x for x in os.listdir(dir_path) if x.endswith('.csv')]

    frame_num = []
    accuracy_rate = []
    molecule_num = []
    for f in files:
        num = f.split('.')[0]
        num = num.split('_')[-1]
        if num.isdigit():
            result = base_calling(os.path.join(dir_path, f), int(num) * 0.95,
                                  exp_type=exp_type, display=display,
                                  correct_pick=correct_pick)
            frame_num.append(int(num))

            result = result[result['confidence'] > minimum_confidence]
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
    df.to_csv(dir_path + '/frame_vs_accuracy_confidence{}.csv'.format(minimum_confidence), index=False)
    print(df)
    return


if __name__ == '__main__':
    #path1 = "G:/time_vs_accuracy/5base/pos6/csv_files"
    #path2 = "G:/time_vs_accuracy/nonComp/nonComp_GapT/csv_files"
    #path3 = "G:/time_vs_accuracy/comp/comp_GapT/csv_files"
    # path4 = "G:/time_vs_accuracy/comp/comp_GapG/csv_files"
    # time_VS_accuracy(path4,
    #                  correct_pick='C', confidence=0.6, exp_type='competitive', display=True)

    #path1 = "G:/accuracy_table/nonComp/8nt_NComp_GAP_A_Seal100nM_GAP_A_localization-1_corrected_neighbour_counting_radius2_inf.csv"
    #path2 = "G:/accuracy_table/nonComp/8nt_GAP_G_Ncomp_GAP_G_localization_corrected_neighbour_counting_radius2_1000.csv"
    #path3 = "G:/accuracy_table/nonComp/8ntGAP_T_Ncomp_seal100nM_Localization_corrected_picasso_bboxes_neighbour_counting_radius2_1000.csv"
    # path4 = "G:/accuracy_table/nonComp/8nt_NComp_GAP_C_Seal100nM_GAP_c_localization_corrected_neighbour_counting_radius2_inf.csv"
    # base_calling(path4,
    #              maximum_length=(1000 * 0.95), exp_type='non-competitive', display=True,
    #              correct_pick='G')

    path1 = "G:/accuracy_table/Comp/8nt_comp_GAP_G_GAP_G_localization_corrected_neighbour_counting_radius2_1200.csv"
    #path2 = "G:/accuracy_table/Comp/8nt_comp_GAP_C_GAP_C_localization_corrected_neighbour_counting_radius2_inf.csv"
    #path3 = "G:/accuracy_table/Comp/GAP_A_8nt_comp_df10_GAP_A_Localization_corrected_neighbour_counting_radius2_inf.csv"
    #path4 = "G:/accuracy_table/Comp/GAP_T_8nt_comp_df10_GAP_T_Localization_corrected_neighbour_counting_radius2_1100.csv"
    base_calling(path1, maximum_length=(1100 * 0.95), exp_type='competitive', display=True,
                 correct_pick='C', save_results=True)


