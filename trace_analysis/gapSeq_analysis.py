import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from trace_analysis_utils import time_series
import multiprocessing


def pick_outlier(binding_params):
    #  ------------------- find outlier ----------------
    binding_params = np.array(binding_params)

    # at least 3 traces have 3 binding events
    if np.sum(binding_params[:, 0] >= 1) < 3:
        outlier = 4
        confidence = 0

    else:
        total_activity = np.sum(binding_params, axis=0)
        scores = binding_params / total_activity
        scores = np.mean(scores, axis=1)
        outlier = np.argmin(scores)


        sorted_scores = np.sort(scores)
        sorted_scores = sorted_scores / sorted_scores[1]

        confidence = (sorted_scores[1] - sorted_scores[0]) / sorted_scores[2:].mean()

    return outlier, confidence


def worker(id, four_traces, penalty=10, mini_size=10, display=False, intensity_threshold=3):
    if display:
        # Create a figure with 4 vertical subplots
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    # ------------get binding parameters for each trace ------------
    binding_params = []
    for i, trace in enumerate(four_traces):
        series = time_series(trace)
        #series.standardize()
        #series.denoise()
        series.PELT_gaussian_analysis(penalty=penalty, mini_size=mini_size)
        #series.PELT_linear_analysis(penalty=penalty, mini_size=mini_size)
        series.get_stage_params()
        series.binary_classify(intensity_threshold=intensity_threshold)
        series.merge_stage()


        binding = series.stage_params[series.stage_params['class'] == 1]
        num_binding = len(binding)
        if num_binding > 0:
            avg_duration = np.sum(binding['end'] - binding['start'])
        else:
            avg_duration = 0

        binding_params.append([num_binding, avg_duration])

        if display:
            # Plot each trace in its respective subplot
            ax = axes[i]
            series.plot(ax=ax)

    # -------------- pick outlier ---------------

    if len(binding_params) == 4:
        outlier, confidence = pick_outlier(binding_params)

    if display:
        axes[0].set_title('ID:{}  pick:{} confidence:{}'.format(id, outlier, np.round(confidence, 2)))
        plt.show()


    return id, outlier, confidence


def io_trace_arrange_csv(file_path, pattern, max_length=np.inf):
    all_traces = pd.read_csv(file_path, skiprows=[2, 3], header=[0, 1])
    A_traces = {}
    T_traces = {}
    C_traces = {}
    G_traces = {}

    # exclude traces that are a flat line
    stds = np.std(all_traces.values, axis=0)
    if np.any(stds == 0):
        all_traces = all_traces.iloc[:, stds != 0]

    for id in all_traces.columns.get_level_values(0).unique():
        subset = all_traces[id]
        if len(subset.columns) == 4:
            for name in subset.columns:
                nucleotide = re.search(pattern, name).group(1)
                trace = subset[name]
                if len(trace) > max_length:
                    trace = trace[:max_length]

                if nucleotide == 'A' or nucleotide == 'a':
                    A_traces[id] = trace
                elif nucleotide == 'T' or nucleotide == 't':
                    T_traces[id] = trace
                elif nucleotide == 'C' or nucleotide == 'c':
                    C_traces[id] = trace
                elif nucleotide == 'G' or nucleotide == 'g':
                    G_traces[id] = trace
                else:
                    raise ValueError('did not match any nucleotide')

    return A_traces, T_traces, C_traces, G_traces



def io_trace_arrange_json(trace_path, pattern, max_length=np.inf):
    all_info = pd.read_json(trace_path)
    all_traces = all_info['data']

    nuc_traces = {}
    index_of_exist = []
    for i in range(4):
        file_name = all_traces.index[i]
        nuc = re.search(pattern, file_name).group(1)

        # this is a list of dictionaries with keys 'picasso_loc' and 'Acceptor' and
        # two only contains NaN values
        one_movie_traces = all_traces.iloc[i]
        one_movie_traces_array = []
        id = 0
        for item in one_movie_traces:
            if len(item) > 0:  # some traces are empty
                trace = item['Acceptor']
                if np.std(trace) != 0:  # some traces are flat lines
                    one_movie_traces_array.append(trace)
                    index_of_exist.append(id)
            id += 1

        one_movie_traces_array = np.array(one_movie_traces_array)
        if one_movie_traces_array.shape[1] > max_length:
            one_movie_traces_array = one_movie_traces_array[:, :max_length]

        nuc_traces[nuc] = one_movie_traces_array

    # -------- only keep the traces that have signal in all 4 movies --------
    inds, counts = np.unique(index_of_exist, return_counts=True)
    index_to_keep = inds[counts == 4]

    # keep the return values the same as the csv version
    A_traces = {}
    T_traces = {}
    C_traces = {}
    G_traces = {}
    for i in index_to_keep:
        A_traces[i] = nuc_traces['A'][i, :]
        T_traces[i] = nuc_traces['T'][i, :]
        C_traces[i] = nuc_traces['C'][i, :]
        G_traces[i] = nuc_traces['G'][i, :]


    return A_traces, T_traces, C_traces, G_traces


# the intensity threshold is in the unit of median intensity std for all detected stages
def GapSeq_analysis(trace_path, pattern, max_length=np.inf, id_list=None,
                    penalty=5, mini_size=5, display=False, intensity_threshold=3):
    if trace_path.endswith('.json'):
        A_traces, T_traces, C_traces, G_traces = io_trace_arrange_json(trace_path, pattern, max_length)

    elif trace_path.endswith('.csv'):
        A_traces, T_traces, C_traces, G_traces = io_trace_arrange_csv(trace_path, pattern, max_length)
    else:
        raise ValueError('file type not supported')

    if id_list is None:
        ids = A_traces.keys()
    else:
        ids = id_list

    args = [(id, (A_traces[id], T_traces[id], C_traces[id], G_traces[id]),
             penalty, mini_size, display, intensity_threshold)
            for id in ids]

    if not display:
        detection_results = []
        with multiprocessing.Pool() as pool:
            for id, outlier, confidence in pool.starmap(worker, args):
                detection_results.append([id, outlier, confidence])

        detection_results = pd.DataFrame(detection_results, columns=['id', 'outlier', 'confidence'])
        detection_results['outlier'] = detection_results['outlier'].replace(
            {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'No signal'})

        if trace_path.endswith('.json'):
            detection_results.to_csv(trace_path.replace('.json', '_detection_results.csv'), index=False)
        else:
            detection_results.to_csv(trace_path.replace('.csv', '_detection_results.csv'), index=False)

        print(detection_results.value_counts(subset=['outlier']))

    else:
        #np.random.shuffle(args)
        for arg in args:
            worker(*arg)


    return





if __name__ == '__main__':
    GapSeq_analysis("H:\jagadish_data\Gap_T_8nt\corrected_movies\GAP_T_8nt_comp_df10_GAP_T_degen100nM_S3A300nM_corrected_NoBG.json",
                    pattern=r'_S3([A-Z])300nM_', display=False, intensity_threshold=3, penalty=2, mini_size=5)

#     params = pd.read_csv("H:/jagadish_data/3 base/base recognition/position 7/GA_seq_comp_13nt_7thpos_interrogation_GAp13nt_L532Exp200_gapseq_detection_results.csv")
#     params = params[params['outlier'] != 'No signal']
#     params = params[params['confidence'] > 0.6]
#     print(len(params))
#     params = params[params['outlier'] != 'G']
#     ids = params['id'].astype(str).to_list()
#
#     GapSeq_analysis("H:/jagadish_data/3 base/base recognition/position 7/GA_seq_comp_13nt_7thpos_interrogation_GAp13nt_L532Exp200_gapseq.csv",
#                     pattern=r'_s7([A-Z])_', display=True, penalty=2, mini_size=5, intensity_threshold=2, id_list=ids)
