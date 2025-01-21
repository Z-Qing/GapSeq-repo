import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def observe_confident_distribution(param_path):
    param = pd.read_csv(param_path)
    param = param[param['Outlier'] != 'No signal']

    for nucleotide in param['Outlier'].unique():
        subset = param[param['Outlier'] == nucleotide]
        plt.hist(subset['Confident Level'], label=nucleotide, bins=50, alpha=0.5)

    plt.legend()
    plt.show()

    return


def accurate_VS_size(param_path, correct_pick):
    param = pd.read_csv(param_path)
    r = []
    num = []
    confidence_thresholds = []
    for c in np.arange(0, 1.0, 0.025):
        subset = param[param['Confident Level'] > c]
        if len(subset) > 0:
            wrong = subset[subset['Outlier'] != correct_pick]
            r.append(1 - len(wrong) / len(subset))
            num.append(len(subset))
            confidence_thresholds.append(c)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(confidence_thresholds, r, '-bo')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='y', colors='blue')
    ax1.yaxis.label.set_color('blue')

    ax2.plot(confidence_thresholds, num, '-ro')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='y', colors='red')
    ax2.yaxis.label.set_color('red')
    plt.show()

    # subset = param[param['Confident Level'] > 0.65]
    # wrong = subset[subset['Outlier'] != 'A']
    # print(1 - len(wrong)/len(subset))
    # print(len(subset))
    return


if __name__ == '__main__':
    path = "H:/jagadish_data/single base/GAP_T_Comp_degenbindingcheck100nM_degen100nM_dex10%seal3A100nM_gapseq_PELT_detection_result.csv"

    #path = "H:/jagadish_data/3 base/base recognition/position 7/GA_seq_comp_13nt_7thpos_interrogation_GAp13nt_L532Exp200_gapseq_PELT_detection_result.csv"

    #path = "H:/jagadish_data/3 base/base recognition/position 6/GAP13nt_position6_comp1uM_degen1uM_buffer20%formamide_GAP13nt_L532L638_Seal6A_degen1uM_gapseq_PELT_detection_result.csv"

    #path = "H:/jagadish_data/3 base/base recognition/position 5/GAP13nt_position5_comp750nM_degen500nM_buffer20%formamide_GAP13nt_L532Exp200_gapseq_PELT_detection_result.csv"

    observe_confident_distribution(path)
    accurate_VS_size(path, correct_pick='A')

