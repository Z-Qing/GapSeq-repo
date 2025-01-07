import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, lfilter
import multiprocess


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def clustering_based_analysis(id, original_signal, segment_points, display=False):
    # Signal preprocessing
    signal = savgol_filter(original_signal, 5, 3, mode='constant')
    signal = np.convolve(signal, np.ones(5) / 5, mode='same')

    intensity_range = np.ptp(signal)
    frame_range = len(signal)
    frame = np.arange(len(signal))

    # Scale the signal and frame for clustering
    scaled_signal = MinMaxScaler().fit_transform(signal.reshape(-1, 1))
    scaled_frame = MinMaxScaler(feature_range=(0, frame_range / intensity_range)).fit_transform(frame.reshape(-1, 1))
    X = np.hstack((scaled_frame, scaled_signal))
    db = DBSCAN(eps=0.02, min_samples=5).fit(X)

    labels = np.unique(db.labels_)
    labels = labels[labels != -1]
    params = []
    for cluster_id in labels:
        cluster_mask = db.labels_ == cluster_id
        median_intensity = np.median(signal[cluster_mask])
        start_frame = frame[cluster_mask][0]
        end_frame = frame[cluster_mask][-1]
        params.append([median_intensity, start_frame, end_frame, cluster_id])

    params = np.array(params)
    #clustering = KMeans(n_clusters=2).fit(params[:, 0].reshape(-1, 1))
    #clustering = AgglomerativeClustering(n_clusters=2).fit(params[:, 0].reshape(-1, 1))
    #params = np.hstack((params, clustering.labels_.reshape(-1, 1)))
    threshold = 1.5 * np.std(signal)
    k_label = params[:, 0] > threshold
    params = np.hstack((params, k_label.reshape(-1, 1)))
    params = pd.DataFrame(params, columns=['median_intensity', 'start_frame', 'end_frame', 'db-label', 'k-label'])
    params['modified'] = False

    high_mean_label = 1

    # Convert data types appropriately
    params['median_intensity'] = params['median_intensity'].astype(float)
    params['start_frame'] = params['start_frame'].astype(int)
    params['end_frame'] = params['end_frame'].astype(int)
    params['db-label'] = params['db-label'].astype(int)
    params['k-label'] = params['k-label'].astype(int)
    params['modified'] = params['modified'].astype(bool)

    # Sort params by start_frame for sequential checking
    params = params.sort_values(by='start_frame').reset_index(drop=True)

    # Initialize refined labels
    max_db_label = params['db-label'].max() + 1
    refined_labels = db.labels_.copy()

    modifications = True
    iteration = 0
    while modifications:
        modifications = False  # Reset modification flag at the start of each iteration

        # Create a copy of params to track changes
        params_prev = params.copy()

        # Step 1: Split clusters if necessary
        params_indices = params.index.tolist()
        for i in params_indices:
            cluster_i = params.loc[i]
            for j in params_indices:
                if i >= j:
                    continue
                cluster_j = params.loc[j]
                # Check for overlap and different k-labels
                if (cluster_j['start_frame'] <= cluster_i['end_frame'] and
                        cluster_j['end_frame'] >= cluster_i['start_frame'] and
                        cluster_i['k-label'] != cluster_j['k-label'] and
                        cluster_i['db-label'] != cluster_j['db-label']):
                    # Overlapping clusters with different k-labels
                    overlap_start = max(cluster_i['start_frame'], cluster_j['start_frame'])
                    overlap_end = min(cluster_i['end_frame'], cluster_j['end_frame'])

                    start_i = cluster_i['start_frame']
                    end_i = cluster_i['end_frame']

                    # First part before overlap
                    if start_i < overlap_start:
                        mask_first_part = (frame >= start_i) & (frame < overlap_start) & \
                                          (refined_labels == cluster_i['db-label'])
                        if np.any(mask_first_part):
                            # Update refined_labels
                            refined_labels[mask_first_part] = cluster_i['db-label']
                            # Update params for the first part
                            params.at[i, 'end_frame'] = overlap_start - 1
                            params.at[i, 'median_intensity'] = np.median(signal[mask_first_part])
                            params.at[i, 'modified'] = True
                    else:
                        # No first part, remove the original cluster_i
                        refined_labels[refined_labels == cluster_i['db-label']] = -1
                        params = params.drop(i)
                        modifications = True
                        break  # Exit inner loop to reprocess params after modifications

                    # Second part after overlap
                    if overlap_end < end_i:
                        mask_second_part = (frame > overlap_end) & (frame <= end_i) & \
                                           (refined_labels == cluster_i['db-label'])
                        if np.any(mask_second_part):
                            # Assign a new DBSCAN label
                            refined_labels[mask_second_part] = max_db_label
                            # Create a new entry in params for the second part
                            new_row = {
                                'median_intensity': np.median(signal[mask_second_part]),
                                'start_frame': overlap_end + 1,
                                'end_frame': end_i,
                                'db-label': max_db_label,
                                'k-label': cluster_i['k-label'],
                                'modified': True
                            }
                            params = params.append(new_row, ignore_index=True)
                            max_db_label += 1

                    # Remove overlap portion from cluster_i
                    mask_overlap = (frame >= overlap_start) & (frame <= overlap_end) & \
                                   (refined_labels == cluster_i['db-label'])
                    refined_labels[mask_overlap] = -1  # Assign as noise or handle appropriately

                    modifications = True
                    print(f"Split cluster {cluster_i['db-label']} at iteration {iteration}")
                    break  # Break inner loop to reprocess params after modifications
            if modifications:
                # Re-sort params after modifications and break to restart the outer loop
                params = params.sort_values(by='start_frame').reset_index(drop=True)
                break

        # Step 2: Combine clusters if necessary
        params_indices = params.index.tolist()
        i = 0
        while i < len(params_indices) - 1:
            idx_i = params_indices[i]
            idx_j = params_indices[i + 1]
            cluster_i = params.loc[idx_i]
            cluster_j = params.loc[idx_j]

            # Check if both clusters belong to the same k-means cluster
            if cluster_i['k-label'] == cluster_j['k-label']:
                # Get the end frame of cluster_i and start frame of cluster_j
                end_i = cluster_i['end_frame']
                start_j = cluster_j['start_frame']

                # Check for any conflicting clusters between cluster_i and cluster_j
                gap_frames = frame[int(end_i + 1): int(start_j)]
                if len(gap_frames) == 0:
                    # No gap, clusters are adjacent or overlapping
                    merge_clusters = True
                else:
                    # Check if any clusters in between belong to different k-label
                    clusters_in_between = params[
                        (params['start_frame'] <= gap_frames[-1]) & (params['end_frame'] >= gap_frames[0])]
                    clusters_in_between = clusters_in_between[
                        ~clusters_in_between['db-label'].isin([cluster_i['db-label'], cluster_j['db-label']])]
                    conflict_clusters = clusters_in_between[clusters_in_between['k-label'] != cluster_i['k-label']]
                    merge_clusters = len(conflict_clusters) == 0

                if merge_clusters:
                    print(
                        f"Merging clusters {cluster_i['db-label']} and {cluster_j['db-label']} at iteration {iteration}")

                    # Update refined_labels for cluster_j to cluster_i's db-label
                    mask_j = refined_labels == cluster_j['db-label']
                    refined_labels[mask_j] = cluster_i['db-label']

                    # Update 'db-label' in params for cluster_j to cluster_i's db-label
                    params.at[idx_j, 'db-label'] = cluster_i['db-label']

                    # Update cluster_i's 'end_frame' and 'start_frame'
                    params.at[idx_i, 'start_frame'] = min(cluster_i['start_frame'], cluster_j['start_frame'])
                    params.at[idx_i, 'end_frame'] = max(cluster_i['end_frame'], cluster_j['end_frame'])

                    # Recalculate 'scaled_mean_intensity' for cluster_i
                    mask_i = refined_labels == cluster_i['db-label']
                    params.at[idx_i, 'median_intensity'] = np.median(signal[mask_i])
                    params.at[idx_i, 'modified'] = True

                    # Remove cluster_j from params
                    params = params.drop(idx_j)
                    params_indices.pop(i + 1)
                    modifications = True
                    # Do not increment i, need to check the new cluster_i with the next cluster
                    continue
            i += 1

        iteration += 1
        # Re-sort params after modifications
        params = params.sort_values(by='start_frame').reset_index(drop=True)

        # update k-labels
        params['k-labels'] = params['median_intensity'] > threshold
        params['k-labels'] = params['k-labels'].astype(int)

        # Check for convergence
        if params.equals(params_prev):
            print("Convergence reached.")
            break

    # Get the outlier and confidence level
    binding_parameters = []
    for k in np.arange(1, len(segment_points)):
        max_frame = segment_points[k]
        min_frame = segment_points[k - 1]
        subset_params = params[(params['start_frame'] >= min_frame) & (params['end_frame'] < max_frame)]

        binding_subset_params = subset_params[subset_params['k-label'] == high_mean_label]
        total_binding_duration = (binding_subset_params['end_frame'] - binding_subset_params['start_frame']).sum()

        binding_event_num = len(binding_subset_params)

        unbinding_subset_params = subset_params[subset_params['k-label'] != high_mean_label]
        avg_binding_intensity = binding_subset_params['median_intensity'].mean() - unbinding_subset_params['median_intensity'].mean()

        binding_parameters.append([avg_binding_intensity, total_binding_duration, binding_event_num])

    binding_parameters = np.array(binding_parameters)
    if np.any(binding_parameters):  # Check if binding_parameters is not all zeros
        binding_parameters = MinMaxScaler().fit_transform(binding_parameters)
        scores = np.mean(binding_parameters, axis=1)
        indices = np.argsort(scores)
        avg_non_outlier_score = scores[indices[2:]].mean()
        second_min_score = scores[indices[1]]
        if avg_non_outlier_score == 0:
            outlier_index = 4
            confident_level = 0
        else:
            outlier_index = indices[0]
            confident_level = (second_min_score - scores[outlier_index]) / avg_non_outlier_score
    else:
        outlier_index = 4
        confident_level = 0

    # Optional visualization
    if display:
        unique_refined_labels = np.unique(refined_labels)
        unique_refined_labels = unique_refined_labels[unique_refined_labels != -1]

        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(original_signal)), original_signal, label='Original Signal')
        plt.plot(np.arange(len(signal)), signal, label='Processed Signal')
        for cluster_id in unique_refined_labels:
            plt.plot(frame[refined_labels == cluster_id], signal[refined_labels == cluster_id], '.', label=f'Cluster {int(cluster_id)}')

        # Highlight clusters with the higher mean intensity
        high_mean_clusters = params[params['k-label'] == high_mean_label]['db-label'].unique()
        for cluster_id in high_mean_clusters:
            plt.plot(frame[refined_labels == cluster_id], signal[refined_labels == cluster_id], 'k-')

        for x in segment_points:
            plt.axvline(x=x, color='r', linestyle='--')

        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        plt.title(id)
        plt.show()

    return id, outlier_index, confident_level


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


def Gap_seq_data_analysis(read_path, pattern=r'seal3([A-Z])100nM', display=True):
    A_traces, T_traces, C_traces, G_traces, length = trace_arrange(read_path, pattern)

    segment_points = [length * i for i in np.arange(5)]

    # get parameter list for multiprocess
    process_params = []
    for id in A_traces.keys():
        trace_set = []
        for trace in [A_traces[id], T_traces[id], C_traces[id], G_traces[id]]:
            trace = trace - min(np.percentile(trace, 10), np.median(trace.iloc[:10]))
            trace_set.append(trace)
        signal = np.concatenate(trace_set, axis=0)
        process_params.append([id, signal, segment_points, display])

    # Release memory for unused data
    del A_traces, T_traces, C_traces, G_traces

    if display:
        for p in process_params:
            clustering_based_analysis(*p)

    elif not display:
        # Multiprocess the clustering based analysis function
        detection_params = []
        with multiprocess.Pool(12) as pool:
            for id, outlier_index, confident_level in pool.starmap(clustering_based_analysis, process_params):
                detection_params.append([id, outlier_index, confident_level])

        detection_params = pd.DataFrame(detection_params, columns=['ID', 'Outlier', 'Confident Level'])
        detection_params['Outlier'].replace({0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'No signal'}, inplace=True)
        print(detection_params.value_counts(subset=['Outlier']))

        save_path = read_path.replace('.csv', '_clustering_detection_result.csv')
        detection_params.to_csv(save_path, index=False)

    else:
        raise ValueError('display should be either True or False')

    return


if __name__ == '__main__':
    Gap_seq_data_analysis("H:/jagadish_data/"
        "GAP_T_Comp_degenbindingcheck100nM_degen100nM_dex10%seal3A100nM_gapseq.csv",
                          pattern=r'seal3([A-Z])100nM', display=False)