import pandas as pd
import multiprocessing
from bocd_utils import time_series_bocd
import numpy as np


def bocd_worker(id, data, display=False):
    trace = time_series_bocd(data)
    trace.bocd()
    trace.PELT_linear_analysis()

    if display:
        trace.plot_posterior(id)

    return id, trace.stage_parameter


def bocd_analysis(path, display=False):
    all_traces = pd.read_csv(path)
    # all_traces = pd.read_csv(path, header=[0, 1], skiprows=[1, 3])
    # all_traces = all_traces.loc[:, (slice(None), 'Acceptor')]

    if display:
        ids = list(all_traces.columns)
        np.random.shuffle(ids)
        for id in ids:
            one_trace = all_traces[id]
            one_trace = one_trace.dropna()
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
    bocd_analysis("G:/CAP_dwellTime_analysis/manuscript_plot/Examples_GapC_GapG_CAP_binding_traces.csv",
                  display=True)



