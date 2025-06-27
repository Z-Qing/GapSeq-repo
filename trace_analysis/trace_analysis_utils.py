import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering


class time_series():
    def __init__(self, time_series):
        self.trace = np.array(time_series)
        self.length = len(time_series)
        self.max = max(time_series)
        self.min = min(time_series)
        self.stage_params = None
        self.bkps = None


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

    def denoise(self):
        self.trace = savgol_filter(self.trace, 21, 3)
        # self.trace = np.convolve(signal, np.ones(20) / 20, mode='same')

        return

    def standardize(self):
        # if self.smooth is None:
        data = self.trace.reshape(-1, 1)
        # else:
        # data = self.smooth.reshape(-1, 1)
        scaler = RobustScaler()
        self.trace = scaler.fit_transform(data).reshape(-1)

        return self.trace


    def PELT_linear_analysis(self, penalty=10, mini_size=10):
        algo = rpt.KernelCPD(kernel='linear', min_size=mini_size).fit(self.trace)
        bkps = algo.predict(pen=penalty)
        bkps.insert(0, 0)

        self.bkps = bkps

        return bkps


    def PELT_gaussian_analysis(self, penalty=10, mini_size=10):
        data = self.trace

        algo = rpt.KernelCPD(kernel="rbf", min_size=mini_size).fit(data)

        bkps = algo.predict(pen=penalty)

        bkps.insert(0, 0)

        self.bkps = bkps

        return bkps


    def plot(self, ax):
        ax.plot(np.arange(self.length), self.trace, label='Original Signal', color='#377eb8')
        if self.stage_params is not None:
            if 'class' in self.stage_params.columns:
                color_map = {0:'#4daf4a', 1:'#e41a1c', 2:'#000000'}
                color_list = [color_map[self.stage_params['class'].iloc[i]] for i in range(len(self.stage_params))]
            else:
                color_list = ['#4daf4a'] * len(self.stage_params)

            previous_intensity = 0
            for i, row in self.stage_params.iterrows():
                ax.hlines(row['intensity'], xmin=row['start'], xmax=row['end'], colors=color_list[i])
                if i > 0:
                    ax.vlines(row['start'], ymin=previous_intensity, ymax=row['intensity'], colors='#984ea3')
                previous_intensity = row['intensity']


    def base_correction(self):
        if self.stage_params is None:
            raise ValueError("Please run a change point detection algorithm first.")

        if np.any(self.stage_params['intensity'] < 0):
            positive = self.stage_params[self.stage_params['intensity'] > 0]
            mini_intensity = positive['intensity'].min()
            self.stage_params['intensity'] = np.where(self.stage_params['intensity'] < 0,
                                                      mini_intensity, self.stage_params['intensity'])

        return


    def intensity_classify(self, intensity_threshold=3):
        if self.stage_params is None:
            raise ValueError("Please run a change point detection algorithm first.")

        if len(self.stage_params) == 1:
            self.stage_params['class'] = 0
            return

        std_list = []
        for i in range(1, len(self.stage_params)):
            std = self.trace[int(self.stage_params['start'][i] + 1):int(self.stage_params['end'][i] - 1)].std()
            std_list.append(std)

        threshold = intensity_threshold * np.median(std_list)
        data = self.stage_params['intensity'].values.reshape(-1, 1)
        cluster = AgglomerativeClustering(linkage='average', distance_threshold=threshold,
                                          n_clusters=None).fit(data)
        labels = cluster.labels_

        cluster_means = {}
        for lbl in np.unique(labels):
            cluster_means[lbl] = data[labels == lbl].mean()

        sorted_labels = sorted(cluster_means, key=cluster_means.get)
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
        new_labels = np.array([label_map[lbl] for lbl in labels])

        self.stage_params['class'] = np.where(new_labels >=2, 2, new_labels)

        return


    def merge_stage(self):
        # in cap binding, we assume there are only three class: 2 binding (dimer) and unbinding
        # when the intensity is too high, we exclude this stage and adjacent
        # binding since they can't provide correct binding duration.
        #class_num = 3
        if self.stage_params is None:
            raise ValueError("Please run a change point detection algorithm first.")

        if 'class' not in self.stage_params.columns:
            raise ValueError("classifying the stages first.")

        # Ensure the DataFrame is sorted by 'start' time
        stage_params = self.stage_params.sort_values(by='start')

        # merge consecutive stages with belong to the same class
        stage_params['group'] = (
                (stage_params['class'] != stage_params['class'].shift())
        ).cumsum()

        # Perform the aggregation to merge stages
        stage_params = stage_params.groupby('group').agg({
            'intensity': 'mean',
            'start': 'min',
            'end': 'max',
            'class': 'first',
            #'merge': 'first',
        }).reset_index(drop=True)

        # Recalculate the real mean intensity using the original signal
        stage_params['intensity'] = stage_params.apply(
            lambda row: np.mean(self.trace[int(row['start'] + 1):int(row['end'] - 1)]), axis=1)

        #stage_params.sort_values(by='start', inplace=True)
        stage_params.reset_index(drop=True, inplace=True)

        self.stage_params = stage_params

        return



