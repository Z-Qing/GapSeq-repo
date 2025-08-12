import pandas as pd
import numpy as np
import os

def sort_traces(trace_path, calling_result_path, min_confidence=0.2):
    calling_results = pd.read_csv(calling_result_path)
    filtered_calling = calling_results.loc[calling_results['confidence'] >= min_confidence]
    to_keep = filtered_calling['ref_index'].to_list()

    traces = pd.read_csv(trace_path)
    intensity_columns_to_keep = np.isin(list(traces.columns), to_keep)
    filtered_traces = traces.loc[:, intensity_columns_to_keep]

    # please note some high confidence molecules have significant number of frames
    # with a negative fluorescence intensities and therefore not in trace file (filtered out)
    ref_index_to_keep = traces.columns.to_numpy()[intensity_columns_to_keep].astype(int)
    filtered_calling = filtered_calling.loc[filtered_calling['ref_index'].isin(ref_index_to_keep)]

    dir_path = os.path.dirname(trace_path)

    for nuc in filtered_calling['calling'].unique():
        nuc_index = filtered_calling.loc[filtered_calling['calling'] == nuc, 'ref_index'].astype(str)
        nuc_intensities = filtered_traces[nuc_index]

        save_path = dir_path + '/Gap{}_traces.csv'.format(nuc)

        nuc_intensities.to_csv(save_path, index=False)

    return


trace_path = "J:/CAP binding/20250709_CAP_library_1base/CAP_library_1base_CAP_binding_2.5nM_corrected_combine_DeepFRET_intensity.csv"
calling_result_path = "J:/CAP binding/20250709_CAP_library_1base/sequencing/CAP_library_1base_library_localization_corrected_neighbour_counting_radius2_inf_base_calling_result.csv"

sort_traces(trace_path, calling_result_path)