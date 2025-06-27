import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trace_analysis_utils import time_series
import multiprocessing
from tifffile import imread

def trace_extraction(ref_locs, mov_path, box_size=3, save=True):

    ref_locs = pd.read_hdf(ref_locs, 'locs')

    movie = imread(mov_path)

    # green_roi = (0, 0, 684, 428)
    # green = movie[:, green_roi[0]:green_roi[2], green_roi[1]:green_roi[3]].astype(np.float32)

    red_roi = (0, 428, 684, 856)
    red = movie[:, red_roi[0]:red_roi[2], red_roi[1]:red_roi[3]]

    traces = pd.DataFrame(np.zeros((red.shape[0], len(ref_locs))), columns=ref_locs.index)
    for i in ref_locs.index:
        pos = np.round(ref_locs.loc[i][['x', 'y']]).astype(int)

        x_start = pos['x'] - box_size//2
        x_end = pos['x'] + box_size//2
        y_start = pos['y'] - box_size//2
        y_end = pos['y'] + box_size//2

        one_trace = red[:, y_start:y_end, x_start:x_end].sum(axis=(1, 2))
        traces[i] = one_trace

    if save:
        traces.to_csv(mov_path.replace('.tif', '_traces.csv'))

    return traces


def worker(id, trace, penalty=3.0, mini_size=10, display=False, intensity_threshold=3.0):
    trace = time_series(trace)
    trace.PELT_gaussian_analysis(penalty=penalty, mini_size=mini_size)
    trace.get_stage_params()
    trace.intensity_classify(intensity_threshold=intensity_threshold)
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


def cap_trace_analysis(path, display=False, intensity_threshold=3.0,
                       penalty=5.0, mini_size=10):
    all_traces = pd.read_csv(path, header=0, skiprows=[1, 2, 3])
    #all_traces = pd.read_csv(path)

    ids = list(all_traces.columns)

    if display:
        for id in ids:
            trace = all_traces[id]
            worker(id, trace, penalty=penalty, mini_size=mini_size, display=True, intensity_threshold=intensity_threshold)
            plt.show()
        return

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
                store.put('index_' + str(key), df)

    return combined


if __name__ == '__main__':
    fitting_result = cap_trace_analysis("Z:/Qing_2/GAPSeq/CAP binding/20250624_CAP_1base_seqN/Median/Result of CAP_1base_seqN_CAP_binding_corrected_BGmedian_gapseq.csv",
                       display=False, intensity_threshold=2.5, penalty=3, mini_size=3)

