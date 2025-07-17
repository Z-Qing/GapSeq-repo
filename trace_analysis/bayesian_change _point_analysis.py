import pandas as pd
import multiprocessing
from bocd_utils import time_series_bocd


def bocd_worker(data, display=False):
    trace = time_series_bocd(data)
    trace.bocd()
    trace.find_changepoints()
    trace.get_stage_parameters()

    if display:

        trace.plot_posterior()

    return



def bocd_analysis(path):
    all_traces = pd.read_csv(path)
    for id in all_traces.columns:
        one_trace = all_traces[id]
        bocd_worker(one_trace, display=True)


    return

if __name__ == "__main__":
    bocd_analysis("G:/CAP_library_2.5nM_library_localization_corrected_first_frame_locs_DeepFRET_intensity.csv")