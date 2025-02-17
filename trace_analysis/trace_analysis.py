import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import RobustScaler
import multiprocessing
from skimage.filters import threshold_otsu
from sklearn.cluster import AgglomerativeClustering

class time_series():
    def __init__(self, time_series):
        self.trace = np.array(time_series)
        self.length = len(time_series)
        self.max = max(time_series)
        self.min = min(time_series)
        self.stage_params = None
        self.bkps = None
        #self.smooth = None


    def __getitem__(self, index):
        return self.trace[index]


    def get_stage_params(self):
        if self.bkps is None:
            raise ValueError("Please run a change point detection algorithm first.")

        starts = self.bkps[:-1]
        ends = self.bkps[1:]
        # Calculate mean intensities for each segment
        intensities = [np.mean(self.trace[start + 1: end - 1]) for start, end in zip(starts, ends)]

        # Combine results into the desired format
        stage_params = np.column_stack([intensities, starts, ends, range(len(starts))])

        stage_params = pd.DataFrame(stage_params, columns=['intensity', 'start', 'end', 'stage']
                                    ).astype({'start': int, 'end': int, 'stage': int})

        self.stage_params = stage_params

        return stage_params


    def PELT_linear_analysis(self, penalty=500000, mini_size=10):
        algo = rpt.KernelCPD(kernel='linear', min_size=mini_size).fit(self.trace)
        bkps = algo.predict(pen=penalty)
        bkps.insert(0, 0)

        self.bkps = bkps

        return bkps


    def denoise(self):
        signal = savgol_filter(self.trace, 21, 3)
        #self.trace = np.convolve(signal, np.ones(20) / 20, mode='same')
        self.smooth = signal

        return

    def standardize(self):
        # if self.smooth is None:
        data = self.trace.reshape(-1, 1)
        # else:
        #data = self.smooth.reshape(-1, 1)
        scaler = RobustScaler()
        self.trace = scaler.fit_transform(data).reshape(-1)

        return self.trace


    def PELT_gaussian_analysis(self, penalty=10, mini_size=10):
        #if self.smooth is None:
        data = self.trace
        # else:
        #     data = self.smooth

        algo = rpt.KernelCPD(kernel="rbf", min_size=mini_size).fit(data)

        bkps = algo.predict(pen=penalty)

        bkps.insert(0, 0)

        self.bkps = bkps

        return bkps


    def plot(self, ax):
        ax.plot(np.arange(self.length), self.trace, label='Original Signal', color='#377eb8')
        # if self.smooth is not None:
        #     ax.plot(np.arange(self.length), self.smooth, label='Smooth Signal', color='#ff7f00')
        if self.stage_params is not None:
            if 'class' in self.stage_params.columns:
                colors_list = ['#e41a1c' if self.stage_params['class'][i] == 1 else '#4daf4a'
                               for i in range(len(self.stage_params))]
            else:
                colors_list = ['#4daf4a'] * len(self.stage_params)

            previous_intensity = 0
            for i, row in self.stage_params.iterrows():
                ax.hlines(row['intensity'], xmin=row['start'], xmax=row['end'], colors=colors_list[i])
                if i > 0:
                    ax.vlines(row['start'], ymin=previous_intensity, ymax=row['intensity'], colors='#984ea3')
                previous_intensity = row['intensity']

        # plt.legend()
        # plt.show()


    def merge_stage(self):
        if self.stage_params is None:
            raise ValueError("Please run a change point detection algorithm first.")

        if 'class' not in self.stage_params.columns:
            raise ValueError("classifying the stages first.")

        # Ensure the DataFrame is sorted by 'start' time
        stage_params = self.stage_params.sort_values(by='start')

        stage_params['group'] = (
                (stage_params['class'] != stage_params['class'].shift())
        ).cumsum()

        # Perform the aggregation to merge stages
        stage_params = stage_params.groupby('group').agg({
            'intensity': 'mean',
            'start': 'min',
            'end': 'max',
            'class': 'first',
        }).reset_index(drop=True)

        # Recalculate the real mean intensity using the original signal
        stage_params['intensity'] = stage_params.apply(
            lambda row: np.mean(self.trace[int(row['start'] + 1):int(row['end'] - 1)]), axis=1)

        stage_params.sort_values(by='start', inplace=True)
        stage_params.reset_index(drop=True, inplace=True)

        self.stage_params = stage_params

        return


    def binary_classify(self):
        if self.stage_params is None:
            raise ValueError("Please run a change point detection algorithm first.")

        if len(self.stage_params) == 1:
            self.stage_params['class'] = 0
            return

        # threshold = threshold_otsu(self.trace)
        # std = min(self.trace[self.trace >= threshold].std(), self.trace[self.trace < threshold].std())
        #
        # threshold = threshold + 3 * std
        # self.stage_params['class'] = (self.stage_params['intensity'] > threshold).astype(int)
        #
        # if np.all(self.stage_params['class'] == 1):
        #     self.stage_params['class'] = 0

        std_list = []
        for i in range(1, len(self.stage_params)):
            std = self.trace[int(self.stage_params['start'][i] + 1):int(self.stage_params['end'][i] - 1)].std()
            std_list.append(std)

        data = self.stage_params['intensity'].values.reshape(-1, 1)
        cluster = AgglomerativeClustering(linkage='average', distance_threshold=2*np.mean(std_list),
                                          n_clusters=None).fit(data)
        labels = cluster.labels_

        cluster_means = {}
        for lbl in np.unique(labels):
            cluster_means[lbl] = data[labels == lbl].mean()

        sorted_labels = sorted(cluster_means, key=cluster_means.get)
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
        new_labels = np.array([label_map[lbl] for lbl in labels])

        self.stage_params['class'] = (new_labels > 0).astype(int)

        return




def io_trace_arrange(file_path, pattern):
    all_traces = pd.read_csv(file_path, skiprows=[2, 3], header=[0, 1])
    A_traces = {}
    T_traces = {}
    C_traces = {}
    G_traces = {}

    for id in all_traces.columns.get_level_values(0).unique():
        subset = all_traces[id]
        if len(subset.columns) == 4:
            for name in subset.columns:
                nucleotide = re.search(pattern, name).group(1)
                if nucleotide == 'A' or nucleotide == 'a':
                    A_traces[id] = subset[name]
                elif nucleotide == 'T' or nucleotide == 't':
                    T_traces[id] = subset[name]
                elif nucleotide == 'C' or nucleotide == 'c':
                    C_traces[id] = subset[name]
                elif nucleotide == 'G' or nucleotide == 'g':
                    G_traces[id] = subset[name]
                else:
                    raise ValueError('did not match any nucleotide')

    return A_traces, T_traces, C_traces, G_traces


def pick_outlier(id, four_traces, penalty=10, mini_size=10, display=False):
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
        series.binary_classify()
        series.merge_stage()

        binding = series.stage_params[series.stage_params['class'] == 1]
        num_binding = len(binding)
        if num_binding > 0:
            avg_duration = np.sum(binding['end'] - binding['start'])
        else:
            avg_duration = 0

        # if num_binding > 2:
        #     pos = (binding['start'] + binding['end']) / 2
        #     count, bins_count = np.histogram(pos, bins=int(series.length / 10), range=(0, series.length))
        #     pdf = count / sum(count)  # the PDF of the histogram using count values
        #     cdf = np.cumsum(pdf)
        #
        #     ideal_CDF = [10 / series.length * i for i in np.arange(1, len(bins_count))]
        #     even = 1 - np.sqrt(mean_squared_error(ideal_CDF, cdf))
        #
        # else:  # when binding event number is less than 3, consider the distribution is not even
        #     even = 0.5

        binding_params.append([num_binding, avg_duration])

        if display:
            # Plot each trace in its respective subplot
            ax = axes[i]
            series.plot(ax=ax)

    #  ------------------- find outlier ----------------
    binding_params = np.array(binding_params)
    #binding_params = RobustScaler().fit_transform(binding_params)

    # at least 3 traces have 3 binding events
    if np.sum(binding_params[:, 0] >= 3) < 3:
        outlier = 4
        confidence = 0

    else:
        total_activity = np.sum(binding_params, axis=0)
        scores = binding_params / total_activity
        scores = np.mean(scores, axis=1)
        outlier = np.argmin(scores)

        unit = np.sort(scores)[1]  # the second smallest score won't be 0
        scores = scores / unit
        avg_left = np.mean(np.delete(scores, outlier))

        confidence = 1 - scores[outlier] / avg_left

    if display:
        axes[0].set_title('ID:{}  pick:{} confidence:{}'.format(id, outlier, np.round(confidence, 2)))
        plt.show()

    return id, outlier, confidence


def GapSeq_analysis(trace_path, pattern, id_list=None, penalty=5, mini_size=5, display=False):
    A_traces, T_traces, C_traces, G_traces = io_trace_arrange(trace_path, pattern)

    if id_list is None:
        ids = A_traces.keys()
    else:
        ids = id_list

    args = [(id, (A_traces[id], T_traces[id], C_traces[id], G_traces[id]), penalty, mini_size, display)
            for id in ids]

    if not display:
        detection_results = []
        with multiprocessing.Pool() as pool:
            for id, outlier, confidence in pool.starmap(pick_outlier, args):
                detection_results.append([id, outlier, confidence])

        detection_results = pd.DataFrame(detection_results, columns=['id', 'outlier', 'confidence'])
        detection_results['outlier'] = detection_results['outlier'].replace(
            {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'No signal'})
        detection_results.to_csv(trace_path.replace('.csv', '_detection_results.csv'), index=False)

        print(detection_results.value_counts(subset=['outlier']))

    else:
        #np.random.shuffle(args)
        for arg in args:
            pick_outlier(*arg)


    return



if __name__ == '__main__':
    params = pd.read_csv("H:\jagadish_data\GAP_A_8nt_comp_df10_GAP_A_Localization_gapseq_detection_results.csv")
    params = params[params['confidence'] > 0.8]
    params = params[params['outlier'] != 'T']
    ids = params['id'].astype(str).to_list()

    GapSeq_analysis("H:\jagadish_data\GAP_A_8nt_comp_df10_GAP_A_Localization_gapseq.csv",
                    pattern=r'S3([A-Z])300nM', display=True, penalty=2, mini_size=10, id_list=ids)




