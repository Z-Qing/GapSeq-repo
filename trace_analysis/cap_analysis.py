import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trace_analysis_utils import time_series
import multiprocessing


def worker(id, trace, penalty=10, mini_size=10, display=False, intensity_threshold=3):
    trace = time_series(trace)
    trace.PELT_gaussian_analysis(penalty=penalty, mini_size=mini_size)
    trace.get_stage_params()
    trace.binary_classify(intensity_threshold=intensity_threshold)
    trace.merge_stage()

    if display:
        ax = plt.subplot(111)
        trace.plot(ax=ax)
        ax.set_title('ID:{}'.format(id))

    binding = trace.stage_params[trace.stage_params['class'] == 1]
    duration = binding['end'] - binding['start']

    unbinding= trace.stage_params[trace.stage_params['class'] == 0]
    blank = unbinding['end'] - unbinding['start']

    return id, duration.values, blank.values, trace.stage_params


def cap_trace_analysis(path, display=False, intensity_threshold=3,
                       penalty=5, mini_size=10):
    all_traces = pd.read_csv(path, header=0, skiprows=[1, 2, 3])
    ids = list(all_traces.columns)

    if display:
        np.random.shuffle(ids)
        for id in ids:
            trace = all_traces[id]
            worker(id, trace, penalty=penalty, mini_size=mini_size, display=True, intensity_threshold=intensity_threshold)
            plt.show()
    else:
        binding_duration = {}
        dark_duration = {}
        fitting_result = {}
        args = [(id, all_traces[id], penalty, mini_size, display, intensity_threshold) for id in ids]
        with multiprocessing.Pool() as pool:
            for id, bd, dd, stage_param  in pool.starmap(worker, args):
                binding_duration[id] = bd
                dark_duration[id] = dd
                fitting_result[id] = stage_param

        dfs = []
        for data, source in zip(
                [binding_duration,  dark_duration],
                ['binding duration',  'dark duration']
        ):
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index = df.index.astype(int)
            df['source'] = source
            dfs.append(df)

        # Concatenate vertically (axis=0) instead of horizontally
        combined = pd.concat(dfs)
        combined = combined.set_index('source', append=True)  # Makes source the second level
        combined = combined.sort_index(level=[0, 1])
        combined.to_csv(path.replace('.csv', '_trace_analysis.csv'), index=True, header=False)

    with pd.HDFStore(path.replace('.csv', '_fitting_result.hdf5'), 'w') as store:
        for key, df in fitting_result.items():
            store.put(key, df)

    return


if __name__ == '__main__':
    cap_trace_analysis("G:/CAP binding/20250427_Gseq1base_CAPbinding2nd/CAP_binding_combined_corrected_bgremove.csv",
                       display=False, intensity_threshold=3, penalty=2, mini_size=5)

