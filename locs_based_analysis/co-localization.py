import pandas as pd
from localization_counting import neighbour_counting
from picasso.io import load_locs


def get_co_localization_rate(ref_path, reanneal_path):
    ref, _ = load_locs(ref_path, 'locs')
    ref = pd.DataFrame.from_records(ref)
    ref = ref.groupby(by='group').mean()

    reanneal, _ = load_locs(reanneal_path, 'locs')
    reanneal = pd.DataFrame.from_records(reanneal)
    reanneal = reanneal.groupby(by='group').mean()

    counting_param = neighbour_counting(ref, reanneal, 'reanneal')
    co_localized = counting_param[counting_param['reanneal'] > 0]

    rate = len(co_localized) / len(counting_param)
    print(rate)

    return


get_co_localization_rate(ref_path="G:/co-localization analysis/test/box_7/eps100_minSample3/13ntGAP_insitu_GAP13_localization_corrected-1_locs_dbscan.hdf5",
                         reanneal_path="G:/co-localization analysis/test/box_7/eps100_minSample3/13ntGAP_insitu_GAP13_reanneal_corrected-1_locs_dbscan.hdf5")