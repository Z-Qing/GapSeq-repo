import pandas as pd
import multiprocess
from scipy.signal import savgol_filter
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.cluster import AgglomerativeClustering
import re


def outlier_detect(original_binding_params, user_thresholds):
    # 1. Basic sanity checks
    if np.sum(original_binding_params == 0) >= 6:
        return 4, 0


    binding_params = original_binding_params.copy()
    total_activity = np.sum(binding_params, axis=0)

    scores = (total_activity - binding_params) / total_activity
    scores = np.sum(scores, axis=1)

    temperature = 0.5
    exp_scores = np.exp(scores / temperature)

    softmax = exp_scores / np.sum(exp_scores)

    outlier_index = np.argmax(softmax)
    most_active_index = np.argmin(softmax)


    ratio = np.clip(original_binding_params[most_active_index] / user_thresholds, 0, 1)
    #mag = ratio.mean()
    #mag = 1.0 - np.prod(1.0 - ratio)
    p = 0.25
    mag = ((ratio ** p).mean()) ** (1 / p)
    #
    # print(mag)
    # print(softmax)

    confidence = mag * softmax[outlier_index]


    return outlier_index, confidence



def change_point_analysis(original_signal, mini_size=5):
    # Smooth the signal using Savitzky-Golay filter and low-pass Butterworth filter
    signal = savgol_filter(original_signal, 11, 5)
    b, a = butter(3, 0.1, btype='low')
    signal = lfilter(b, a, signal)
    signal = np.convolve(signal, np.ones(10) / 10, mode='same')

    # Detect change points using the PELT algorithm with linear kernel
    algo = rpt.KernelCPD(kernel='linear', min_size=mini_size).fit(signal)
    bkps = algo.predict(pen=250000)
    bkps.insert(0, 0)

    starts = bkps[:-1]
    ends = bkps[1:]
    # Calculate mean intensities for each segment
    intensities = [np.mean(signal[start:end]) for start, end in zip(starts, ends)]

    # Combine results into the desired format
    stage_params = np.column_stack([intensities, starts, ends, range(len(starts))])

    stage_params = pd.DataFrame(stage_params, columns=['intensity', 'start', 'end', 'stage'])


    return stage_params, signal


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
    stage_params.sort_values(by='start', inplace=True)

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
        'nucleotide': 'first'
    }).reset_index(drop=True)

    # Recalculate the real mean intensity using the original signal
    stage_params['intensity'] = stage_params.apply(
        lambda row: signal[int(row['start']):int(row['end'])].mean(), axis=1)

    return stage_params



def baseline_correction(stage_params, smoothed_signal, movie_length, stds,
                        nucleotide_sequence):
    # ---------------------- Merge stages ----------------------
    #k_label = intensity_classification_kmeans(stage_params['intensity'].to_numpy().reshape(-1, 1))
    k_label = intensity_classification_Aggo(stage_params['intensity'].to_numpy().reshape(-1, 1), num_class=4)
    stage_params['k_label'] = k_label

    # this process is to merge stages that have the same k_label
    stage_params = merge_stage(stage_params, smoothed_signal)

    # ---------------------- Correct intensity in smoothed signal ----------------------
    # get the minimum intensity of one stage from each segmentation and update the
    # intensity column and signal accordingly
    baseline = np.nan
    base_corrected_signal = smoothed_signal.copy()
    for i in np.arange(4):
        n = nucleotide_sequence[i]

        mask = stage_params['nucleotide'] == n
        subset = stage_params[mask]

        for k in np.arange(0, 4):
            same_k_stage = subset[subset['k_label'] == k]
            durations = same_k_stage['end'] - same_k_stage['start']
            total_duration = durations.sum()

            # if the total duration of a certain k_label is too small it can be noise
            duration_mask = durations > movie_length // 10
            if np.any(duration_mask):
                baseline = same_k_stage.loc[duration_mask, 'intensity'].min()
                break

            # if lack of a single stage long enough to be considered as base
            if total_duration > movie_length // 5:
                baseline = np.average(same_k_stage['intensity'], weights=durations)
                break

        if np.isnan(baseline):
            raise ValueError('No baseline found for nucleotide {}'.format(n))

        stage_params.loc[mask, 'intensity'] -= baseline
        start = movie_length * i
        end = movie_length * (i + 1)
        base_corrected_signal[start: end] = smoothed_signal[start: end] - baseline

    # ---------------------- Assign stages into 2 classes on and off ----------------------
    # k-means clustering again after intensity correction and update k-labels
    ratio = stds.max() / stds.min()
    #print(ratio)
    if ratio >= 4:
        k_label = intensity_classification_Aggo(stage_params['intensity'].to_numpy().reshape(-1, 1), num_class=3)
        k_label = (k_label > 0).astype(int)
        stage_params['k_label'] = k_label

    elif 2 < ratio < 4:
        threshold = stage_params['intensity'].max() / 3
        k_label = (stage_params['intensity'] >= threshold).astype(int)
        stage_params['k_label'] = k_label

    else: #  all or none of the nucleotides are binding
        possible_k_label = intensity_classification_Aggo(stage_params['intensity'].to_numpy().reshape(-1, 1), num_class=3)
        stage_params['k_label'] = possible_k_label
        possible_intensity = stage_params.groupby('k_label')['intensity'].mean()
        possible_intensity_diff = np.diff(possible_intensity.sort_values().to_numpy())
        # print(possible_intensity_diff)
        # print(stds)
        # if the difference between the mean intensity of the two classes is too large
        if np.any(possible_intensity_diff > 3 * stds.max()):
            stage_params['k_label'] = (possible_k_label > 0).astype(int)
        else:
            stage_params['k_label'] = (stage_params['intensity'] > 1.5 * stds.max()).astype(int)


    # threshold_column = stage_params['nucleotide'].apply(lambda x: stds[nucleotide_sequence.index(x)])
    # new_k_label = (stage_params['intensity'] > threshold_column).astype(int)
    # stage_params['k_label'] = new_k_label

    # merge again
    stage_params = merge_stage(stage_params, base_corrected_signal)

    # ---------------------- Correct k_label ----------------------
    # in case baseline correction or classification is wrong
    # that only binding events are detected. We switch it to un-binding events
    for i in np.arange(4):
        mask = stage_params['nucleotide'] == nucleotide_sequence[i]
        subset = stage_params[mask]

        if len(subset) == 0:
            raise ValueError('No stage found for nucleotide {}'.format(nucleotide_sequence[i]))

        if subset['k_label'].min() > 0:
            stage_params.loc[mask, 'k_label'] = 0


    return stage_params, base_corrected_signal


def PELT_trace_fitting(id, original_signal_list, display=False):
    movie_length = len(original_signal_list[0])
    nucleotide_sequence = ['A', 'T', 'C', 'G']
    intensity_stds = np.array([np.std(s) for s in original_signal_list])

    if np.any(intensity_stds == 0):
        return id, 4, 0

    # ------------- Change point analysis creating stage_params ----------------------
    stage_params = []
    smoothed_signal = []
    for i in np.arange(4):
        signal_one_nuc = original_signal_list[i]
        # change point analysis
        stage_param_one_nuc, smoothed_signal_one_nuc = change_point_analysis(signal_one_nuc)
        stage_param_one_nuc['nucleotide'] = nucleotide_sequence[i]
        stage_param_one_nuc['start'] = stage_param_one_nuc['start'] + movie_length * i
        stage_param_one_nuc['end'] = stage_param_one_nuc['end'] + movie_length * i

        stage_params.append(stage_param_one_nuc)
        smoothed_signal.append(smoothed_signal_one_nuc)

    # Concatenate the results
    stage_params = pd.concat(stage_params, axis=0)
    smoothed_signal = np.concatenate(smoothed_signal, axis=0)

    # baseline correction and stage merging
    stage_params, signal = baseline_correction(stage_params, smoothed_signal, movie_length, intensity_stds, nucleotide_sequence)


    # -------------Calculate binding parameters for each nucleotide --------------
    binding_params = pd.DataFrame(np.zeros((4, 3)), columns=['total_duration', 'event_num', 'binding_intensity'],
                                  index=nucleotide_sequence)
    for i in np.arange(4):
        n = nucleotide_sequence[i]
        subset = stage_params[stage_params['nucleotide'] == n]

        binding_subset = subset[subset['k_label'] > 0]
        unbinding_subset = subset[subset['k_label'] == 0]

        if len(binding_subset) == 0:
            binding_params.loc[n] = [0, 0, 0]
            continue

        total_binding_duration = (binding_subset['end'] - binding_subset['start']).sum()
        binding_event_num = len(binding_subset)

        avg_binding_intensity = (np.average(binding_subset['intensity'], weights=(binding_subset['end'] - binding_subset['start'])) -
                                 np.average(unbinding_subset['intensity'], weights=(unbinding_subset['end'] - unbinding_subset['start'])))

        binding_params.loc[n] = [total_binding_duration, binding_event_num, avg_binding_intensity]

    # detect outliers
    outlier, confident = outlier_detect(binding_params.to_numpy(), np.array([500, 10, max(intensity_stds)]))

    # Display the results
    if display:
        original_signals = np.concatenate(original_signal_list, axis=0)
        plt.plot(np.arange(len(original_signals)), original_signals, label='Original Signal')
        plt.plot(np.arange(len(signal)), signal, label='Corrected Signal')

        #plt.hlines(xmin=0, xmax=len(signal), y=ref_int, color='black', linestyle='--', label='Threshold')

        colors = ['green', 'red']
        pre_intensity = 0
        for i in np.arange(len(stage_params)):
            current_intensity = stage_params['intensity'].iloc[i]
            plt.vlines(stage_params['start'].iloc[i], ymin=pre_intensity, ymax=current_intensity)
            c = colors[stage_params['k_label'].iloc[i]]
            plt.hlines(stage_params['intensity'].iloc[i], xmin=stage_params['start'].iloc[i], xmax=stage_params['end'].iloc[i], color=c)

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
    length = len(all_traces)
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

    return A_traces, T_traces, C_traces, G_traces, length


def Gapseq_data_analysis(read_path, pattern=r'', display=False, save=True, id_list=None):
    A_traces, T_traces, C_traces, G_traces, length = trace_arrange(read_path, pattern)

    if id_list is None:
        id_list = A_traces.keys()

    process_params =[]
    for id in id_list:
        trace_set = [A_traces[id], T_traces[id], C_traces[id], G_traces[id]]
        process_params.append((id, trace_set,  display))

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
        detection_params['Outlier'] = detection_params['Outlier'].replace({0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'No signal'})
        print(detection_params.value_counts(subset=['Outlier']))

        if save:
            save_path = read_path.replace('.csv', '_PELT_detection_result.csv')
            detection_params.to_csv(save_path, index=False)

    else:
        raise ValueError('display should be either True or False')


    return


if __name__ == '__main__':
    # path = "H:/jagadish_data/5 base/position 7/GAP-seq_5ntseq_position7_dex10%formamide2_gapseq.csv"
    # pattern = r'_seal7([A-Z])4uM_.'

    # path = "H:/jagadish_data/3 base/base recognition/position 7/GA_seq_comp_13nt_7thpos_interrogation_GAp13nt_L532Exp200_gapseq.csv"
    # pattern = r'_s7([A-Z])_'

    # path = "H:/jagadish_data/single base/GA_seq_comp_13nt_7thpos_interrogation_GAp13nt_L532Exp200_gapseq.csv"
    # pattern = r'_s7([A-Z])_'

    # path = "H:/jagadish_data/3 base/base recognition/position 5/GAP13nt_position5_comp750nM_degen500nM_buffer20%formamide_GAP13nt_L532Exp200_gapseq.csv"
    # pattern = r'_comp5([A-Za-z])_'

    # path = "H:/jagadish_data/3 base/base recognition/position 6/GAP13nt_position6_comp1uM_degen1uM_buffer20%formamide_GAP13nt_L532L638_Seal6A_degen1uM_gapseq.csv"
    # pattern = r'_Seal6([A-Z])_'

    # path = "H:/jagadish_data/5 base/position 5/5nt_13GAP_pos5_dex20%__seqeucing_S5A_5uM_degen2uM_gapseq.csv"
    # pattern = r'_S5([A-Z])_'

    # path = "H:/jagadish_data/5 base/position 6/5nt_13GAP_pos6_dex15%__form20%_seqeucing1_degen2uM_s6A4uM_gapseq.csv"
    # pattern = r'_s6([A-Za-z])4uM'

    # path = "H:/jagadish_data/5 base/position 6/5nt_13GAP_pos6_dex15%__form20%_seqeucing2_degen2uM_s6A4uM_gapseq.csv"
    # pattern = r'_s6([A-Za-z])4uM'

    # path = "H:/jagadish_data/5 base/position 8/GAP-seq_5ntseq_position8_dex13%formamide7%_GAPlocalizationL532Exp200_gapseq.csv"
    # pattern = r'_Seal8([A-Z])4uM_'

    path = "H:/jagadish_data/5 base/position 9/5nt_13GAP_pos9_dex15%__form20%_seqeucing_degen2uM_seal9A4uM_gapseq.csv"
    pattern = r'_seal9([A-Z])4uM'

    # params = pd.read_csv("H:/jagadish_data/5 base/position 5/5nt_13GAP_pos5_dex20%__seqeucing_S5A_5uM_degen2uM_gapseq_PELT_detection_result.csv")
    # params = params[params['Outlier'] != 'No signal']
    # params = params[params['Confident Level'] > 0.5]
    # ids = params['ID'].astype(str).to_list()

    Gapseq_data_analysis(path, pattern=pattern, display=False, save=True)

