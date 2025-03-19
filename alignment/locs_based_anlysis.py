from picasso_utils import one_channel_movie
from locs_based_analysis_preAligned_transcription_movie import neighbour_counting
import picasso.render as _render
from picasso.io import save_locs
from picasso.imageprocess import rcc
import re
import os
import pandas as pd
from pystackreg import StackReg
import numpy as np



def locs_based_analysis(movie_path_list, ref_movie_path, pattern, transformation_matrix,
                        box_size=2, gpu=True, roi=None, ref_roi=None, save_hdf5=True):

    ''' This function corrects the position of the movies in the movie_path_list based on the reference movie
    Then it calculates the overlap between the reference movie and the movies in the movie_path_list'''

    # ------- prepare the reference locs -------
    ref = one_channel_movie(ref_movie_path, ref_roi, frame_range=0)
    ref.lq_fitting(GPU=gpu, box=5, min_net_gradient=400)
    ref.overlap_prevent(box_radius=box_size)

    _, ref_image = _render.render(ref.locs, ref.info)

    if save_hdf5:
        save_locs(ref_movie_path.replace('.tif', '.hdf5'), ref.locs, ref.info)

    nuc_locs = {}
    for movie_path in movie_path_list:
        # ----------------------- drift correction ----------
        mov = one_channel_movie(movie_path, roi)
        mov.lq_fitting(GPU=gpu, min_net_gradient=1000, box=5)
        mov.drift_correction()
        mov.overlap_prevent(box_radius=box_size)

        nuc = re.search(pattern, os.path.basename(movie_path)).group(1)

        # ----------------------- channel alignment ----------
        #if transformation_matrix is None:
        #mov_image = mov.movie[0, :, :]
        # _, mov_image = _render.render(mov.locs, mov.info)
        # sr = StackReg(StackReg.RIGID_BODY)
        # transformation_matrix = sr.register(ref_image, mov_image)
        # print(transformation_matrix)

        h, w = ref.movie.shape[1:]
        center = np.array([w/2, h/2])
        pos = pd.DataFrame(mov.locs[['x', 'y']])
        pos_homogeneous = np.hstack([pos - center, np.ones((pos.shape[0], 1))])
        pos_homogeneous = np.dot(transformation_matrix, pos_homogeneous.T).T
        pos_homogeneous = pos_homogeneous[:, :2] + center

        mov.locs.x = pos_homogeneous[:, 0]
        mov.locs.y = pos_homogeneous[:, 1]
        nuc_locs[nuc] = mov.locs[['x', 'y']].copy()


        if save_hdf5:
            save_locs(movie_path.replace('.tif', '.hdf5'), mov.locs, mov.info)

    # ----------------------- neighbour counting ----------------------
    ref_locs = ref.locs[['x', 'y']].copy()
    total_params = []
    for nuc in nuc_locs.keys():
        param = neighbour_counting(ref_locs, nuc_locs[nuc], nuc)
        total_params.append(param)

    total_params = pd.concat(total_params, axis=1)
    total_params.to_csv(os.path.dirname(ref_movie_path) + '/neighbour_counting.csv', index=True)


    return


def sort_files(dir_path):
    movie_files = [x for x in os.listdir(dir_path) if x.endswith('.tif')]
    ref = [x for x in movie_files if 'Localizaiton' in x or 'Localization' in x]

    if len(ref) != 1:
        raise ValueError('there should be one and only one localization file')
    else:
        ref = ref[0]

    movie_files.remove(ref)
    mov_list = [os.path.join(dir_path, x) for x in movie_files]


    return os.path.join(dir_path, ref), mov_list



if __name__ == "__main__":
    ref_roi = [0, 0, 684, 428]  # green channel # Note that two NIMs have different width!!!!
    roi = [0, 428, 684, 856]  # red channel

    ref, mov_list = sort_files("G:/GAP30A")
    mat = [[0.9982168695689115, 0.002647757967641371, -3.194892980677258],
           [-0.002490229429364575, 0.9994192246786789, -4.464671817808321],
           [4.32733735129356e-06, 1.2548130752315316e-07, 1.0]]

    locs_based_analysis(mov_list, ref, pattern=r'_3([A-Z])_', transformation_matrix=mat,
                        roi=roi, ref_roi=ref_roi, save_hdf5=True, gpu=True)






