import pandas as pd
import multiprocess
from scipy.signal import savgol_filter
import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
import re


def outlier_detect(binding_params, ref_intensity_change):
    """
    Detects the least active outlier time series and calculates confidence.

    Parameters:
    - binding_params: ndarray with shape (4, 3), where each row contains:
        [total_binding_duration, binding_event_nums, avg_binding_intensity].
    - std: Standard deviation of the data for setting thresholds dynamically.

    Returns:
    - outlier: Index of the outlier time series (0-3 or 4 if no data).
    - confidence: Confidence score (0-1), scaled by activity magnitude.
    """
    # Define user-set thresholds dynamically
    user_thresholds = np.array([100, 5, ref_intensity_change])

    if not np.any(binding_params):
        return 4, 0

    binding_params = np.array(binding_params)
    max_binding_param = binding_params.max(axis=0)
    if np.any(max_binding_param == 0):
        return 4, 0

    invert_params = binding_params.max(axis=0) - binding_params
    normalized_params = invert_params / max_binding_param

    # Weighted scores for each feature
    weights = np.array([1/3, 1/3, 1/3])  # Adjust weights based on feature importance
    scores = np.dot(normalized_params, weights)

    indices = np.argsort(scores)  # Sort indices by scores
    outlier_index = indices[-1]   # Identify the least active outlier (lowest score)

    # Calculate average score of the remaining traces
    average_other_score = scores[indices[:-1]].mean()

    # Confidence level calculation
    #if (1 - average_other_score) != 0:
    confident = (scores[outlier_index] - average_other_score) / (1 - average_other_score)
    # else:
    #     confident = 0

    # Activity magnitude adjustment based on the second least active trace
    second_least_active = binding_params[indices[-2], :]  # Parameters of the second least active trace

    # Calculate activity magnitude
    if np.all(second_least_active >= user_thresholds):
        activity_magnitude = 1
    else:
        activity_magnitude = np.clip((second_least_active / user_thresholds), 0, 1).mean()

    # Adjust confidence with activity magnitude
    confident *= activity_magnitude

    return outlier_index, confident


def PELT_trace_fitting(id, original_signal, segment_points, display=False):
    mini_size = 5
    # signal = savgol_filter(original_signal, 11, 5)
    # b, a = butter(3, 0.1, btype='low')
    # signal = lfilter(b, a, signal)
    # signal = np.convolve(signal, np.ones(10) / 10, mode='same')

    # subtract the baseline for each trace
    signal = []
    for i in np.arange(len(segment_points) - 1):
        sub_signal = original_signal[segment_points[i]:segment_points[i + 1]]
        sub_signal = savgol_filter(sub_signal, 11, 5)
        b, a = butter(3, 0.1, btype='low')
        sub_signal = lfilter(b, a, sub_signal)
        sub_signal = np.convolve(sub_signal, np.ones(10) / 10, mode='same')
        print(np.percentile(sub_signal, 15))
        sub_signal = sub_signal - np.percentile(sub_signal, 10)
        signal.append(sub_signal)
        #signal[segment_points[i]:segment_points[i + 1]] = (signal[segment_points[i]:segment_points[i + 1]] -
                                          #np.percentile(signal[segment_points[i]:segment_points[i + 1]], 10))
    #signal = np.where(signal > 0, signal, 0)
    signal = np.concatenate(signal, axis=0)

    # Detect change points using the PELT algorithm with linear kernel
    algo = rpt.KernelCPD(kernel='linear', min_size=mini_size).fit(signal)
    bkps = algo.predict(pen=250000)
    bkps = np.append(bkps, segment_points[:-1])
    bkps.sort()

    diffs = np.diff(bkps)
    # Create a mask where differences are smaller than 5
    small_diff_mask = diffs < mini_size
    # Identify breakpoints that overlap with segment_points
    segment_mask = np.isin(bkps[:-1], segment_points)
    # Use the mask to decide which breakpoints to keep
    replace_mask = small_diff_mask & ~segment_mask  # Replace only if not in segment_points
    # Replace the values where needed
    bkps[:-1] = np.where(replace_mask, bkps[1:], bkps[:-1])
    # Remove duplicates caused by replacements
    bkps = np.unique(bkps)

    if len(bkps) <= 2:
        return id, 4, 0

    starts = bkps[:-1]
    ends = bkps[1:]
    # Calculate mean intensities for each segment
    intensities = [np.mean(signal[start:end]) for start, end in zip(starts, ends)]
    # Combine results into the desired format
    stage_params = np.column_stack([intensities, starts, ends, range(len(starts))])

    # threshold = 1.25 * np.std(signal)
    # k_label = np.clip((stage_params[:, 0] // threshold), 0, 2)
    kmeans = KMeans(n_clusters=3).fit(stage_params[:, 0].reshape(-1, 1))
    # Sort centroids and create a mapping
    centroids = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centroids)  # Indices of centroids from smallest to largest
    label_mapping = {sorted_indices[i]: i for i in range(3)}
    k_label = np.array([label_mapping[label] for label in kmeans.labels_])

    stage_params = np.hstack((stage_params, k_label.reshape(-1, 1)))
    stage_params = pd.DataFrame(stage_params, columns=['intensity', 'start', 'end', 'stage', 'k_label'])
    stage_params['k_label'] = stage_params['k_label'].astype(int)

    # Recalculate segment membership based on updated segment points
    stage_params['segment'] = pd.cut(
        stage_params['start'],
        bins=segment_points,
        labels=False,
        right=False
    )

    # Perform merging within each segment separately
    # Merge stages: Create a group based on both binding and merge condition
    stage_params['group'] = (
            (stage_params['k_label'] != stage_params['k_label'].shift()) |
            (stage_params['segment'] != stage_params['segment'].shift())
    ).cumsum()

    # Perform the aggregation to merge stages
    stage_params = stage_params.groupby('group').agg({
        'intensity': 'mean',
        'start': 'min',
        'end': 'max',
        'k_label': 'first'
    }).reset_index(drop=True)

    # # Update the 'stage' column
    # stage_params['stage'] = np.arange(len(stage_params))

    # Ensure the DataFrame is sorted by 'start' time
    stage_params.sort_values(by='start', inplace=True)

    # Recalculate the real mean intensity using the original signal
    stage_params['intensity'] = stage_params.apply(
        lambda row: signal[int(row['start']):int(row['end'])].mean(), axis=1)

    # Update the 'stage' column
    stage_params['stage'] = np.arange(len(stage_params))

    # # Ensure the DataFrame is sorted by 'start' time
    # stage_params.sort_values(by='start', inplace=True)

    binding_params = []
    for i in np.arange(len(segment_points) - 1):
        max_frame = segment_points[i + 1]
        min_frame = segment_points[i]
        #subset = stage_params[(stage_params['start'] >= min_frame) & (stage_params['end'] < max_frame)]
        subset = stage_params[(stage_params['end'] > min_frame) & (stage_params['start'] < max_frame)]

        binding_subset = subset[subset['k_label'] >= 1]
        unbinding_subset = subset[subset['k_label'] == 0]

        if len(binding_subset) == 0:
            binding_params.append([0, 0, 0])
            continue

        total_binding_duration = (binding_subset['end'] - binding_subset['start']).sum()
        binding_event_num = len(binding_subset)
        binding_intensity_mean = binding_subset['intensity'].mean()

        if len(unbinding_subset) == 0:
            binding_params.append([total_binding_duration, binding_event_num, binding_intensity_mean])
            continue

        unbinding_intensity_mean = unbinding_subset['intensity'].mean()
        avg_binding_intensity = binding_intensity_mean - unbinding_intensity_mean
        binding_params.append([total_binding_duration, binding_event_num, avg_binding_intensity])

    ref_int = np.std(signal)
    outlier, confident = outlier_detect(binding_params, ref_int)

    if display:
        plt.plot(np.arange(len(original_signal)), original_signal, label='Original Signal')
        plt.plot(np.arange(len(signal)), signal, label='Smoothed Signal')

        plt.hlines(xmin=0, xmax=len(signal), y=ref_int, color='black', linestyle='--', label='Threshold')

        colors = ['green', 'yellow', 'red']
        pre_intensity = 0
        for i in np.arange(len(stage_params)):
            current_intensity = stage_params['intensity'].iloc[i]
            plt.vlines(stage_params['start'].iloc[i], ymin=pre_intensity, ymax=current_intensity)
            c = colors[stage_params['k_label'].iloc[i]]
            plt.hlines(stage_params['intensity'].iloc[i], xmin=stage_params['start'].iloc[i], xmax=stage_params['end'].iloc[i], color=c)

            pre_intensity = current_intensity

        for i in np.arange(len(segment_points)):
            plt.axvline(x=segment_points[i], linestyle='--', color='black')

        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        plt.title(id)
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
                if nucleotide == 'A':
                    A_traces[id] = subset[name]
                elif nucleotide == 'T':
                    T_traces[id] = subset[name]
                elif nucleotide == 'C':
                    C_traces[id] = subset[name]
                elif nucleotide == 'G':
                    G_traces[id] = subset[name]
                else:
                    raise ValueError('did not match any nucleotide')

    return A_traces, T_traces, C_traces, G_traces, length


def Gapseq_data_analysis(read_path, pattern=r'', display=False, save=True, id_list=None):

    A_traces, T_traces, C_traces, G_traces, length = trace_arrange(read_path, pattern)

    segment_points = [length * i for i in np.arange(5)]
    if id_list is None:
        id_list = A_traces.keys()

    process_params =[]
    for id in id_list:
        trace_set = []
        for trace in [A_traces[id], T_traces[id], C_traces[id], G_traces[id]]:
            trace = trace - np.percentile(trace, 15)
            trace_set.append(trace)
        signal = np.concatenate(trace_set, axis=0)
        process_params.append((id, signal, segment_points, display))

    # Release memory for unused data
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
    #path = "H:/jagadish_data/5 base/position 7/GAP-seq_5ntseq_position7_dex10%formamide2_gapseq.csv"
    path = "H:/jagadish_data/3 base/base recognition/position 7/GA_seq_comp_13nt_7thpos_interrogation_GAp13nt_L532Exp200_gapseq.csv"
    Gapseq_data_analysis(path,
                   pattern=r'_s7([A-Z])_', display=True, save=True)

