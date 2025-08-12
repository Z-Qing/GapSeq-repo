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
        unbinding_time = []
        dict = {}
        param_list = [(id, all_traces[id]) for id in all_traces.columns]
        with multiprocessing.Pool() as pool:
            for id, stage in pool.starmap(bocd_worker, param_list):
                dict[id] = stage
                if len(stage) >= 3:
                    # remove the stage the beginning and end as they don't
                    # provide accurate time
                    filtered_stage = stage.iloc[1: -1]
                    binding = filtered_stage[filtered_stage['class'] != 0]
                    unbinding = filtered_stage[filtered_stage['class'] == 0]

                    if len(binding) > 0:
                        dwell_time.extend((binding['end'] - binding['start']).to_list())

                    if len(unbinding) > 0:
                        unbinding_time.extend((unbinding['end'] - unbinding['start']).to_list())


        save_path = path.replace('.csv', '_bocd_dwellTime.csv')
        df = pd.Series(dwell_time, name='dwellTime')
        df.to_csv(save_path, index=False)

        save_path = path.replace('.csv', '_bocd_unbindingTime.csv')
        df = pd.Series(unbinding_time, name='unbindingTime')
        df.to_csv(save_path, index=False)

        save_path = path.replace('.csv', '_bocd_results.npy')
        np.save(save_path, dict)

    return

if __name__ == "__main__":
    bocd_analysis("J:/CAP binding/20250713_CAP_library_1baseNNN/GapG_traces.csv",
                  display=False)