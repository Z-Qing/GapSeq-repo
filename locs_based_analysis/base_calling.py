import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def model_func(t, A, K):
    return A * np.exp(K * t)

def fit_exp_nonlinear(t, y):
    A_guess = y[0]
    K_guess = -0.01  # Initial decay rate guess (negative since we expect decay)

    opt_parms, parm_cov = curve_fit(model_func, t, y, maxfev=1000, p0=(A_guess, K_guess))
    A, K = opt_parms
    return A, K


def threshold_selection(data, bin_size=10, n_decay_length=3):
    data = data[data > 0]
    counts, bin_edges = np.histogram(data, bins=np.arange(0, data.max() + bin_size, bin_size))
    positions = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Calculate first derivative
    first_deriv = np.gradient(counts, positions)
    global_min_idx = np.argmin(first_deriv)
    global_min_x = positions[global_min_idx]

    mask = positions >= global_min_x
    # Linear Fit (Note that we have to provide the y-offset ("C") value!!
    A, K = fit_exp_nonlinear(positions[mask], first_deriv[mask])

    fit_y = model_func(positions[mask], A, K)

    fig, ax = plt.subplots(2, 1)
    ax[0].bar(positions, counts, width=bin_size)

    ax[1].plot(positions[mask], fit_y, '--', label='fitted exponential decay')
    ax[1].plot(positions, first_deriv, label='first derivative')
    ax[1].legend(loc='best')
    plt.show()

    transition_x = np.round(n_decay_length* np.abs(1/K) + global_min_x)
    print('decay length is {}'.format(np.abs(1/K)))
    print('the threshold is {}'.format(transition_x))

    return transition_x


def competitive_selection(param, threshold):
    b = np.sum(param.to_numpy() > threshold, axis=1)
    b = b >= 3
    param = param.loc[b]

    choice = param.idxmin(axis=1)

    sorted = param.to_numpy()
    sorted.sort(axis=1)
    diff = sorted[:, 1] - sorted[:, 0] + 1

    diff = diff / np.percentile(diff, 90)
    diff = np.clip(diff, 0, 1)

    return choice, diff


def non_competitive_selection(param, threshold):
    b = np.sum(param.to_numpy() > threshold, axis=1)
    b = b >= 1
    param = param.loc[b]

    choice = param.idxmax(axis=1)

    sorted = param.to_numpy()
    sorted.sort(axis=1)
    diff = sorted[:, -1] - sorted[:, -2]  + 1 # avoid negative inf

    diff = diff / np.percentile(diff, 90) #diff.max()
    diff = np.clip(diff, 0, 1)

    return choice, diff


def base_calling(path, maximum_length, exp_type, correct_pick=None, threshold=None,
                 bin_width=5, display=False, save_results=False):
    param = pd.read_csv(path, index_col=0)
    #remove fiducial markers
    param = param.loc[~(param.max(axis=1) > maximum_length)]

    # ----------------- threshold selection ------------------------------
    if type(threshold) == int or type(threshold) == float:
        transition_point = threshold
    else:
        locs_counts = param.to_numpy().flatten()
        transition_point = threshold_selection(locs_counts, bin_size=bin_width)


    # ----------------- confidence VS accuracy rate plot -------------------
    if exp_type == 'competitive':
        choice, confidence = competitive_selection(param, transition_point)
    elif exp_type == 'non-competitive':
        choice, confidence = non_competitive_selection(param, transition_point)
    else:
        raise ValueError

    results = param.loc[choice.index].copy()
    results['calling'] = choice
    results['confidence'] = confidence
    results['calling'].replace({'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}, inplace=True)
    if save_results:
        results.to_csv(path.replace('.csv', '_base_calling_result.csv'), index=True)


    if display and isinstance(correct_pick, str):
        fig, ax = plt.subplots(2, 1)

        thresholds = []
        accuracy_rate = []
        molecule_number = []
        for t in np.arange(0, 1, 0.1):
            selected_choice = results.loc[results['confidence'] > t]
            if len(selected_choice) == 0:
                break
            else:
                summary = selected_choice['calling'].value_counts()
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

    return results



def time_VS_accuracy(dir_path, correct_pick, minimum_confidence, exp_type,
                     display=True, threshold=None):
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
                                  correct_pick=correct_pick, threshold=threshold)
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
    path = ("Z:/Jagadish_new/GAP-seq_method/3nt base sequencing/20250914_3baseseq_pos6/processed/"
            "3baseseq_pos6_GAP13_loclaization_picasso_bboxes_neighbour_counting_radius2_inf.csv")
    base_calling(path, maximum_length=(1200 * 0.9), exp_type='competitive', display=True,
                  correct_pick='G', save_results=True, threshold=None)




