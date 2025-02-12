import pandas as pd
import multiprocess
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import re
from scipy.stats import linregress
from scipy.signal import savgol_filter


def non_competitive_outlier_detect(original_binding_params, user_thresholds):
    if np.all(original_binding_params == 0):
        return 4, 0

    binding_params = original_binding_params.copy()
    total_activity = np.sum(binding_params, axis=0)

    if np.any(total_activity == 0):
        return 4, 0

    scores = binding_params / total_activity
    # scores = MinMaxScaler().fit_transform(scores)
    scores = np.sum(scores, axis=1)

    temperature = 0.5
    exp_scores = np.exp(scores / temperature)

    softmax = exp_scores / np.sum(exp_scores)

    outlier_index = np.argmax(softmax)

    ratio = np.clip(original_binding_params[outlier_index] / user_thresholds, 0, 1)
    #mag = ratio.mean()
    # mag = 1.0 - np.prod(1.0 - ratio)
    p = 3
    mag = ((ratio ** p).mean()) ** (1 / p)

    # print(mag)
    # print(softmax)

    confidence = mag * softmax[outlier_index]

    return outlier_index, confidence


def outlier_detect(original_binding_params, user_thresholds):
    # 1. Basic sanity checks
    if np.sum(original_binding_params == 0) > 3:
        return 4, 0

    binding_params = original_binding_params.copy()
    total_activity = np.sum(binding_params, axis=0)

    scores = (total_activity - binding_params) / total_activity
    # scores = MinMaxScaler().fit_transform(scores)
    scores = np.sum(scores, axis=1)

    temperature = 0.5
    exp_scores = np.exp(scores / temperature)

    softmax = exp_scores / np.sum(exp_scores)

    outlier_index = np.argmax(softmax)
    most_active_index = np.argmin(softmax)

    ratio = np.clip(original_binding_params[most_active_index] / user_thresholds, 0, 1)
    # mag = ratio.mean()
    # mag = 1.0 - np.prod(1.0 - ratio)
    p = 3
    mag = ((ratio ** p).mean()) ** (1 / p)

    # print(mag)
    # print(softmax)

    confidence = mag * softmax[outlier_index]

    return outlier_index, confidence


def slope_correction(original_signal):
    # Fourier Transform of the signal
    fft_signal = np.fft.fft(original_signal)
    # Frequency bins
    freqs = np.fft.fftfreq(len(original_signal), d=1)

    # Apply the low-pass filter: set high frequencies to zero
    fft_signal[np.abs(freqs) > 0.01] = 0

    # Inverse Fourier Transform to return to time domain
    smoothed_signal = np.fft.ifft(fft_signal).real

    derivative = np.diff(smoothed_signal)

    algo = rpt.KernelCPD(kernel='linear', min_size=200).fit(derivative)
    bkps = algo.predict(pen=10)
    bkps = [0] + [p + 1 for p in bkps]
    #print(bkps)

    spike_indices = np.where(np.abs(smoothed_signal - smoothed_signal.mean()) > np.std(smoothed_signal))[0]
    # combined_bkps = np.sort(np.unique(np.concatenate([bkps, spike_indices])))
    # print(combined_bkps)

    used_bkps = []
    corrected_signal = original_signal.copy()
    for i in range(len(bkps) - 1):
        start, end = bkps[i], bkps[i + 1]
        segment_indices = np.arange(start, end)

        # Identify valid subsegments (excluding spike regions)
        spike_mask = np.isin(segment_indices, spike_indices)
        valid_subsegments = np.split(segment_indices, np.where(spike_mask)[0])
        valid_subsegments = [seg for seg in valid_subsegments if len(seg) > 100]

        # Perform slope correction within each valid subsegment
        for subsegment in valid_subsegments:
            valid_indices = subsegment
            signal_piece = smoothed_signal[valid_indices]
            original_signal_piece = original_signal[valid_indices]

            # Perform linear regression
            slope, intercept, r, p, se = linregress(valid_indices, signal_piece)
            #print(start, end, slope, p)
            # Apply correction only if slope is significant
            if np.abs(slope) > 0.01 and p < 0.05:
                used_bkps.extend([valid_indices[0], valid_indices[-1]])
                fitted_line = slope * valid_indices + intercept

                # if slope > 0:
                #     corrected_signal[valid_indices] = original_signal_piece - fitted_line + fitted_line[-1]
                # else:
                corrected_signal[valid_indices] = original_signal_piece - fitted_line + fitted_line[0]


    return corrected_signal, used_bkps



def change_point_analysis(original_signal, mini_size=5):
    #signal, slope_change_points = slope_correction(original_signal)
    signal = savgol_filter(original_signal, 11, 5)
    signal = np.convolve(signal, np.ones(20) / 20, mode='same')

    # Detect change points using the PELT algorithm with linear kernel
    algo = rpt.KernelCPD(kernel='linear', min_size=mini_size).fit(signal)
    bkps = algo.predict(pen=250000)
    bkps.insert(0, 0)
    # print(bkps)

    starts = bkps[:-1]
    ends = bkps[1:]
    # Calculate mean intensities for each segment
    intensities = [np.mean(signal[start:end]) for start, end in zip(starts, ends)]

    # Combine results into the desired format
    stage_params = np.column_stack([intensities, starts, ends, range(len(starts))])

    stage_params = pd.DataFrame(stage_params, columns=['intensity', 'start', 'end', 'stage']
                                ).astype({'start': int, 'end': int, 'stage': int})

    return stage_params, signal #, slope_change_points


def intensity_classification_Aggo(intensities, num_class=4):
    clustering = AgglomerativeClustering(n_clusters=num_class).fit(intensities)
    labels = clustering.labels_

    # Map the labels to the original labels so that the cluster
    # with higher mean intensity has a higher label
    cluster_means = {}
    for lbl in np.unique(labels):
        cluster_means[lbl] = intensities[labels == lbl].mean()

    sorted_labels = sorted(cluster_means, key=cluster_means.get)
    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    new_labels = np.array([label_map[lbl] for lbl in labels])

    return new_labels


def merge_stage(stage_params, signal):
    # Ensure the DataFrame is sorted by 'start' time
    stage_params = stage_params.sort_values(by='start')

    stage_params['group'] = (
            (stage_params['k_label'] != stage_params['k_label'].shift()) |
            (stage_params['nucleotide'] != stage_params['nucleotide'].shift())
    ).cumsum()

    # Perform the aggregation to merge stages
    stage_params = stage_params.groupby('group').agg({
        'intensity': 'mean',
        'start': 'min',
        'end': 'max',
        'k_label': 'first',
        'nucleotide': 'first',
    }).reset_index(drop=True)

    # Recalculate the real mean intensity using the original signal
    stage_params['intensity'] = stage_params.apply(
        lambda row: signal[int(row['start']):int(row['end'])].mean(), axis=1)

    stage_params.sort_values(by='start', inplace=True)
    stage_params.reset_index(drop=True, inplace=True)

    return stage_params


# this function is for signal from one nucleotide
def baseline_correction(original_stage_params, signal, movie_length):
    stage_params = original_stage_params.copy()
    corrected_signal = signal.copy()


    durations = original_stage_params['end'] - original_stage_params['start']
    longest_stage = original_stage_params.loc[durations.idxmax()]
    longest_stage_std = np.std(signal[longest_stage['start']:longest_stage['end']])

    # this process is to merge stages that have the same k_label
    stage_params = merge_stage(stage_params, signal)

    # find the baseline after slope correction
    baseline = np.nan
    for k in np.arange(0, 4):
        same_k_stage = stage_params[stage_params['k_label'] == k]
        durations_same_k = same_k_stage['end'] - same_k_stage['start']
        total_duration_same_k = durations_same_k.sum()

        # if the total duration of a certain stage is too small it can be noise
        duration_mask_same_k = durations_same_k > movie_length // 10
        if np.any(duration_mask_same_k):
            baseline = same_k_stage.loc[duration_mask_same_k, 'intensity'].min()
            break

        # if lack of a single stage long enough to be considered as base
        if total_duration_same_k > movie_length // 10 or len(same_k_stage) >= 3:
            baseline = np.average(same_k_stage['intensity'], weights=durations_same_k)
            break

    if np.isnan(baseline):
        raise ValueError('No baseline found')
    else:
        stage_params['intensity'] -= baseline
        corrected_signal -= baseline


    return stage_params, corrected_signal, longest_stage_std


def binary_classification(stage_params, movie_length, signals, nucleotide_sequence):
    k_label = intensity_classification_Aggo(stage_params['intensity'].to_numpy().reshape(-1, 1), num_class=4)
    stage_params['k_label'] = k_label

    # ---------------------- Correct intensity in smoothed signal ----------------------
    # get the minimum intensity of one stage from each segmentation and update the
    # intensity column and signal accordingly
    corrected_stage_params = []
    corrected_signal = []
    long_stage_std_list =[]
    for i in np.arange(4):
        mask = stage_params['nucleotide'] == nucleotide_sequence[i]
        subset_param = stage_params[mask]
        subset_signal = signals[i]

        corrected_subset_parma, subset_corrected_signal, long_stage_std\
            = baseline_correction(subset_param, subset_signal, movie_length)

        corrected_stage_params.append(corrected_subset_parma)
        corrected_signal.append(subset_corrected_signal)
        long_stage_std_list.append(long_stage_std)

    corrected_stage_params = pd.concat(corrected_stage_params, axis=0, ignore_index=True)

    # ---------------------- Assign stages into 2 classes on and off ----------------------
    threshold_column = 3 * corrected_stage_params['nucleotide'].apply(lambda x: long_stage_std_list[nucleotide_sequence.index(x)])
    threshold = 1/3.3 * corrected_stage_params['intensity'].max()
    threshold_column = np.where(threshold_column < threshold, threshold, threshold_column)

    k_label = (corrected_stage_params['intensity'] > threshold_column).astype(int)
    corrected_stage_params['k_label'] = k_label
    # merge again
    # stage_params = merge_stage(stage_params, base_corrected_signal)

    # ---------------------- Correct k_label ----------------------
    # in case baseline correction or classification is wrong
    # that only binding events are detected. We switch it to un-binding events
    for i in np.arange(4):
        mask = corrected_stage_params['nucleotide'] == nucleotide_sequence[i]
        subset = corrected_stage_params[mask]

        if len(subset) == 0:
            raise ValueError('No stage found for nucleotide {}'.format(nucleotide_sequence[i]))

        if subset['k_label'].min() > 0:
            corrected_stage_params.loc[mask, 'k_label'] = 0

    return corrected_stage_params, corrected_signal


def PELT_trace_fitting(id, original_signal_list, comp_exp, display=False):
    movie_length = len(original_signal_list[0])
    nucleotide_sequence = ['A', 'T', 'C', 'G']
    intensity_stds = np.array([np.std(s) for s in original_signal_list])

    if np.any(intensity_stds == 0):
        return id, 4, 0

    # ------------- Change point analysis creating stage_params ----------------------
    stage_params = []
    smoothed_signal_list = []
    for i in np.arange(4):
        signal_one_nuc = original_signal_list[i]
        # change point analysis
        stage_param_one_nuc, corrected_signal = change_point_analysis(signal_one_nuc.to_numpy())
        stage_param_one_nuc['nucleotide'] = nucleotide_sequence[i]
        stage_params.append(stage_param_one_nuc)
        smoothed_signal_list.append(corrected_signal)

    # Concatenate the results
    stage_params = pd.concat(stage_params, axis=0)

    # find the on and off states
    corrected_stage_params, corrected_signal = binary_classification(stage_params, movie_length, smoothed_signal_list,
                                                                    nucleotide_sequence)

    # -------------Calculate binding parameters for each nucleotide --------------
    binding_params = pd.DataFrame(np.zeros((4, 3)), columns=['total_duration', 'event_num', 'binding_intensity'],
                                  index=nucleotide_sequence)
    for i in np.arange(4):
        n = nucleotide_sequence[i]
        subset = corrected_stage_params[corrected_stage_params['nucleotide'] == n]

        binding_subset = subset[subset['k_label'] > 0]
        unbinding_subset = subset[subset['k_label'] == 0]

        if len(binding_subset) == 0:
            binding_params.loc[n] = [0, 0, 0]
            continue

        total_binding_duration = (binding_subset['end'] - binding_subset['start']).sum()
        binding_event_num = len(binding_subset)

        avg_binding_intensity = (
                    np.average(binding_subset['intensity'], weights=(binding_subset['end'] - binding_subset['start'])) -
                    np.average(unbinding_subset['intensity'],
                               weights=(unbinding_subset['end'] - unbinding_subset['start'])))

        binding_params.loc[n] = [total_binding_duration, binding_event_num, avg_binding_intensity]

    # detect outliers
    if comp_exp:
        outlier, confident = outlier_detect(binding_params.to_numpy(), np.array([500, 10, 2 * min(intensity_stds)]))
    else:
        outlier, confident = non_competitive_outlier_detect(binding_params.to_numpy(),
                                                            np.array([500, 10, 2 * min(intensity_stds)]))

    # Display the results
    if display:
        original_signals = np.concatenate(original_signal_list, axis=0)
        plt.plot(np.arange(len(original_signals)), original_signals, label='Original Signal')
        corrected_signals = np.concatenate(corrected_signal, axis=0)
        plt.plot(np.arange(len(corrected_signals)), corrected_signals, label='Corrected Signal')

        fake_start_frame_column = corrected_stage_params['nucleotide'].apply(
            lambda x: movie_length * nucleotide_sequence.index(x))

        colors = ['green', 'red']
        pre_intensity = 0
        for i in np.arange(len(corrected_stage_params)):
            current_intensity = corrected_stage_params['intensity'].iloc[i]
            fake_start = fake_start_frame_column.iloc[i] + corrected_stage_params['start'].iloc[i]
            fake_end = fake_start_frame_column.iloc[i] + corrected_stage_params['end'].iloc[i]

            plt.vlines(fake_start, ymin=pre_intensity, ymax=current_intensity)
            c = colors[corrected_stage_params['k_label'].iloc[i]]
            plt.hlines(corrected_stage_params['intensity'].iloc[i], xmin=fake_start, xmax=fake_end, color=c)

            pre_intensity = current_intensity

        for x in [movie_length * i for i in np.arange(1, 4)]:
            plt.axvline(x=x, linestyle='--', color='black')


        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        plt.title('ID :{}, confidence: {}, picked: {}'.
                  format(id, np.round(confident, 2), outlier))
        plt.show()

    return id, outlier, confident


def trace_arrange(file_path, pattern):
    all_traces = pd.read_csv(file_path, skiprows=[2, 3], header=[0, 1])
    A_traces = {}
    T_traces = {}
    C_traces = {}
    G_traces = {}

    for id in all_traces.columns.get_level_values(0).unique():
        subset = all_traces[id]
        if len(subset.columns) == 4:
            for name in subset.columns:
                nucleotide = re.search(pattern, name).group(1)
                if nucleotide == 'A' or nucleotide == 'a':
                    A_traces[id] = subset[name]
                elif nucleotide == 'T' or nucleotide == 't':
                    T_traces[id] = subset[name]
                elif nucleotide == 'C' or nucleotide == 'c':
                    C_traces[id] = subset[name]
                elif nucleotide == 'G' or nucleotide == 'g':
                    G_traces[id] = subset[name]
                else:
                    raise ValueError('did not match any nucleotide')

    return A_traces, T_traces, C_traces, G_traces


def Gapseq_data_analysis(read_path, pattern=r'', comp_exp=True, display=False, save=True, id_list=None):
    A_traces, T_traces, C_traces, G_traces = trace_arrange(read_path, pattern)

    if id_list is None:
        id_list = A_traces.keys()

    process_params = []
    for id in id_list:
        trace_set = [A_traces[id], T_traces[id], C_traces[id], G_traces[id]]
        process_params.append((id, trace_set, comp_exp, display))

    # Release memory for data that won't be used anymore
    del A_traces, T_traces, C_traces, G_traces

    if display is True:
        np.random.shuffle(process_params)
        for p in process_params:
            PELT_trace_fitting(*p)

    elif display is False:
        detection_params = []
        with multiprocess.Pool(12) as pool:
            for id, outlier, confident in pool.starmap(PELT_trace_fitting, process_params):
                detection_params.append([id, outlier, confident])

        detection_params = pd.DataFrame(detection_params, columns=['ID', 'Outlier', 'Confident Level'])
        detection_params['Outlier'] = detection_params['Outlier'].replace(
            {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'No signal'})
        print(detection_params.value_counts(subset=['Outlier']))

        if save:
            save_path = read_path.replace('.csv', '_PELT_detection_result.csv')
            detection_params.to_csv(save_path, index=False)

    else:
        raise ValueError('display should be either True or False')

    return


if __name__ == '__main__':
    # path = "H:/jagadish_data/single base/GAP_T_Comp_degenbindingcheck100nM_degen100nM_dex10%seal3A100nM_gapseq.csv"
    # pattern = r'seal3([A-Z])100nM'

    # path = "H:/jagadish_data/3 base/base recognition/position 7/GA_seq_comp_13nt_7thpos_interrogation_GAp13nt_L532Exp200_gapseq.csv"
    # pattern = r'_s7([A-Z])_'

    # path = "H:/jagadish_data/3 base/base recognition/position 6/GAP13nt_position6_comp1uM_degen1uM_buffer20%formamide_GAP13nt_L532L638_Seal6A_degen1uM_gapseq.csv"
    # pattern = r'_Seal6([A-Z])_'

    # path = "H:/jagadish_data/3 base/base recognition/position 5/GAP13nt_position5_comp750nM_degen500nM_buffer20%formamide_GAP13nt_L532Exp200_gapseq.csv"
    # pattern = r'_comp5([A-Za-z])_'

    # param = pd.read_csv("H:/jagadish_data/3 base/base recognition/position 6/GAP13nt_position6_comp1uM_degen1uM_buffer20%formamide_GAP13nt_L532L638_Seal6A_degen1uM_gapseq_PELT_detection_result.csv")
    # param = param[param['Outlier'] != 'No signal']
    # param = param[param['Outlier'] != 'C']
    # param = param[param['Confident Level'] > 0.5]
    # ids = param['ID'].astype(str)

    # path = "H:/Jagadish_data/non_complementary/GAP_30T_NonCoomp_seal100nM_1_Localization_gapseq.csv"
    # pattern = r'_seal3([A-Z])_'

    path = "H:\jagadish_data\GAP_A_8nt_comp_df10_GAP_A_Localization_gapseq.csv"
    pattern = r'S3([A-Z])300nM'

    Gapseq_data_analysis(path, pattern=pattern, comp_exp=True, display=False, save=True)