"""============================================================================

Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For algorithm details, see

    Adams & MacKay 2007
    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

For Bayesian inference details about the Gaussian, see:

    Murphy 2007
    "Conjugate Bayesian analysis of the Gaussian distribution"
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

A part of code is adapted from https://github.com/gwgundersen/bocd/

============================================================================"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
import ruptures as rpt
import pandas as pd


class time_series_bocd():
    def __init__(self, data):
        self.trace = data
        self.T = len(data)

        self.stage_parameter = None
        self.R = np.zeros((self.T + 1, self.T + 1))
        self.pmean = np.zeros(self.T + 1)
        self.pvar = np.zeros(self.T + 1)
        self.based_corrected_pmean = np.zeros(self.T + 1)
        self.cps = []

    def bocd(self):
        hazard = 1 / self.T
        # More robust prior estimation
        varx = np.var(self.trace)
        mean0 = np.median(self.trace)  # Using median for robustness

        # Estimate var0 using interquartile range (more robust than variance)
        iqr = np.percentile(self.trace, 75) - np.percentile(self.trace, 25)
        var0 = (iqr / 1.34) ** 2  # Convert IQR to variance estimate

        model = GaussianUnknownMean(mean0, var0, varx)

        log_R = -np.inf * np.ones((self.T + 1, self.T + 1))
        log_R[0, 0] = 0
        pmean = np.empty(self.T)
        pvar = np.empty(self.T)
        log_message = np.array([0])
        log_H = np.log(hazard)
        log_1mH = np.log(1 - hazard)

        for t in np.arange(1, self.T + 1):
            x = self.trace[t - 1]

            # Model predictions with smoothing
            weights = np.exp(log_R[t - 1, :t])
            weights = weights / weights.sum()  # Normalize
            pmean[t - 1] = np.sum(weights * model.mean_params[:t])
            pvar[t - 1] = np.sum(weights * model.var_params[:t])

            log_pis = model.log_pred_prob(t, x)
            log_growth_probs = log_pis + log_message + log_1mH
            log_cp_prob = logsumexp(log_pis + log_message + log_H)

            new_log_joint = np.append(log_cp_prob, log_growth_probs)
            log_R[t, :t + 1] = new_log_joint - logsumexp(new_log_joint)

            model.update_params(t, x)
            log_message = new_log_joint

        self.R = np.exp(log_R)
        self.pmean = pmean
        self.pvar = pvar
        return


    def PELT_linear_analysis(self, penalty=50, mini_size=1):

        data = self.pmean

        # PELT change point detection
        algo = rpt.KernelCPD(kernel='linear', min_size=mini_size).fit(data)
        bkps = algo.predict(pen=penalty)
        bkps.insert(0, 0)

        starts = bkps[:-1]
        ends = bkps[1:]
        self.cps = bkps[1:-1]

        intensities = [np.median(data[start: end]) for start, end in zip(starts, ends)]

        stage_params = np.column_stack([intensities, starts, ends, range(len(starts))])
        stage_params = pd.DataFrame(stage_params, columns=['intensity', 'start', 'end', 'stage']
                                    ).astype({'start': int, 'end': int, 'stage': int})

        stage_params['class'] = 0
        #print(stage_params)
        if len(starts) > 1:
            threshold = 2 * np.median(np.sqrt(self.pvar)) + np.percentile(self.pmean, 15)
            stage_params['class'] = (stage_params['intensity'] > threshold).astype(int)
            # ------------------- merge consecutive stages in the same class-------------------------
            # merge consecutive stages with belong to the same class
            stage_params['group'] = (
                (stage_params['class'] != stage_params['class'].shift())
            ).cumsum()

            # Perform the aggregation to merge stages
            stage_params = stage_params.groupby('group').agg({
                'intensity': 'first',
                'start': 'min',
                'end': 'max',
                'class': 'first',
                # 'merge': 'first',
            }).reset_index(drop=True)

            # Recalculate the real mean intensity using the original signal
            stage_params['intensity'] = stage_params.apply(
                lambda row: np.median(data[int(row['start']):int(row['end'])]), axis=1)

            # stage_params.sort_values(by='start', inplace=True)
            stage_params.reset_index(drop=True, inplace=True)

        self.stage_parameter = stage_params
        print(stage_params)
        self.based_corrected_pmean = data

        return



    def plot_posterior(self, title):
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax1, ax2 = axes

        # Plot the raw time series data
        ax1.plot(np.arange(0, self.T), self.trace, 'b-', alpha=0.5, label='Raw data')
        ax1.set_xlim([0, self.T])
        ax1.margins(0)
        ax1.set_ylabel('Value')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot predictions with uncertainty
        ax1.plot(np.arange(0, self.T), self.pmean, 'k-', linewidth=2, label='Predicted mean')
        #ax1.plot(np.arange(0, self.T), self.based_corrected_pmean, 'k-', linewidth=2, label='Base corrected')
        _std = np.sqrt(self.pvar)
        ax1.fill_between(np.arange(0, self.T),
                         self.pmean - _std,
                         self.pmean + _std,
                         color='gray', alpha=0.2, label='± std')

        # Define a color palette for different classes
        class_colors = {0: 'green', 1: 'orange'}

        for _, row in self.stage_parameter.iterrows():
            start = int(row['start'])
            end = int(row['end'])
            class_id = int(row['class'])

            # Plot the segment
            ax1.axvspan(start, end, facecolor=class_colors[class_id], alpha=0.2)


        # Plot changepoints
        for cp in self.cps:
            ax1.axvline(cp, c='red', ls='--', alpha=0.7, linewidth=1)

        ax1.legend(loc='upper right')
        ax1.set_title(title)

        # Plot the run length posterior
        im = ax2.imshow(np.rot90(self.R), aspect='auto', cmap='gray_r',
                   norm=LogNorm(vmin=0.0001, vmax=1))
        ax2.set_xlim([0, self.T])

        current_ticks = ax2.get_yticks()
        current_labels = [int(tick) for tick in current_ticks]  # Convert to integers if needed
        # Reverse the labels
        reversed_labels = current_labels[::-1]
        # Set fixed ticks and labels to avoid the warning
        ax2.yaxis.set_major_locator(ticker.FixedLocator(current_ticks))
        ax2.set_yticklabels(reversed_labels)

        ax2.margins(0)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Run length')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="2.5%", pad=0.1)  # adjust size/pad
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('P(run)', labelpad=1)

        plt.tight_layout()
        plt.show()


class GaussianUnknownMean:

    def __init__(self, mean0, var0, varx):
        """Initialize model.

        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1 / var0])

    def log_pred_prob(self, t, x):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params = self.prec_params + (1 / self.varx)
        self.prec_params = np.append([1 / self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params = (self.mean_params * self.prec_params[:-1] + \
                           (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1. / self.prec_params + self.varx


