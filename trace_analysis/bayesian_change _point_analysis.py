import pandas as pd
import multiprocessing
from bocd_utils import time_series_bocd
import numpy as np

def bocd_worker(id, data, display=False):
    trace = time_series_bocd(data)
    trace.bocd()
    trace.find_changepoints()
    trace.get_stage_parameters(n_threshold=1.0)

    if display:
        trace.plot_posterior(id)

    return id, trace.stage_params



def bocd_analysis(path, display=False):
    all_traces = pd.read_csv(path)

    if display:
        for id in all_traces.columns:
            one_trace = all_traces[id]
            bocd_worker(id, one_trace, display=True)

    else:
        dwell_time = []
        dit = {}
        param_list = [(id, all_traces[id]) for id in all_traces.columns]
        with multiprocessing.Pool() as pool:
            for id, stage in pool.starmap(bocd_worker, param_list):
                dit[id] = stage
                binding = stage[stage['class'] !=0]
                if len(binding) > 0:
                    dwell_time.extend((binding['end'] - binding['start']).to_list())

        save_path = path.replace('.csv', '_bocd_dwellTime.csv')
        df = pd.Series(dwell_time, name='dwellTime')
        df.to_csv(save_path, index=False)

        save_path = path.replace('.csv', '_bocd_results.npy')
        np.save(save_path, dit)

    return

if __name__ == "__main__":
    bocd_analysis("J:/PELT_bocd_inner2_outer5_sum_intensity_GapG/seal G/"
                  "8nt_NComp_GAP_G_Seal100nM_GAP_G_seal3G_100nM_corrected_DeepFRET_intensity.csv",
                  display=True)