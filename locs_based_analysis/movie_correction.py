import numpy as np
from picasso.imageprocess import rcc
import picasso.render as _render
from tifffile import imwrite, imread
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pystackreg import StackReg
from pystackreg.util import to_uint16
from scipy.ndimage import shift
from picasso_utils import one_channel_movie



def prepare_two_channel_movie(movie_path, gradient_1=400, gradient_2=400, box_1=5, box_2=5,
                              roi=None, gpu=True):
    movie = imread(movie_path)
    h, w = movie.shape[1], movie.shape[2]

    if roi is None:
        rio_1 = [0, 0, h, w//2]
        rio_2 = [0, w//2, h, w]

    else:
        rio_1 = roi[0]
        rio_2 = roi[1]

    channel_1 = one_channel_movie(movie_path, roi=rio_1)
    channel_2 = one_channel_movie(movie_path, roi=rio_2)

    if gpu:
        channel_1.lq_gpu_fitting(gradient_1, box_1)
        channel_2.lq_gpu_fitting(gradient_2, box_2)

    else:
        channel_1.lq_cpu_fitting(gradient_1, box_1)
        channel_2.lq_cpu_fitting(gradient_2, box_2)


    if channel_1.info[0]['Frames'] >= 200:
        channel_1.drift_correction(gpu)
        channel_2.drift_correction(gpu)

    return channel_1, channel_2



def process_frame(frame, transform_mat, sr):
    """
    Process a single frame (CPU-bound task).
    """
    if np.any(frame):
        return to_uint16(sr.transform(frame, tmat=transform_mat))
    else:
        return frame

def process_frame_chunk(frame_chunk, transform_mat, sr):
    """
    Process a chunk of frames using multithreading.
    """
    with ThreadPoolExecutor() as thread_pool:
        # Process all frames in the chunk in parallel using threads
        results = list(thread_pool.map(
            lambda frame: process_frame(frame, transform_mat, sr),
            frame_chunk
        ))
    return results

def stackreg_channel_alignment(mov, ref=None, transfer_matrix=None, num_processes=None):
    """
    Align all frames in `mov` using hybrid parallelism.
    """
    sr = StackReg(StackReg.RIGID_BODY)

    if transfer_matrix is None:
        mov_image = mov[0, :, :]
        transform_mat = sr.register(ref, mov_image)
    else:
        transform_mat = transfer_matrix

    # Split the frames into chunks for multiprocessing
    num_frames = mov.shape[0]
    num_processes = num_processes or multiprocessing.cpu_count()
    chunk_size = num_frames // num_processes
    frame_chunks = [mov[i:i + chunk_size] for i in range(0, num_frames, chunk_size)]

    # Use multiprocessing to process chunks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_results = pool.starmap(
            process_frame_chunk,
            [(chunk, transform_mat, sr) for chunk in frame_chunks]
        )

    # Combine the results from all chunks
    aligned_mov = np.concatenate(chunk_results, axis=0)

    return aligned_mov



def position_correction_fiducial(movie_path_list, ref_movie_path, roi=None, ref_roi=None,
                                 gpu=True, save=True, channel_align_method='RCC'):
    ''' This function do alignment between different green channels using fiducial markers.
    And the fiducial markers can be detected easily in green channel but not red channel.
    The locs in green and red channels of transcription movie should be co-localized. Thus,
    the transformation matrix between green and red channel are acquired from there. '''

    channel_1, channel_2 = prepare_two_channel_movie(ref_movie_path, roi=ref_roi, gpu=gpu)

    _, image_1 = _render.render(channel_1.locs, channel_1.info)
    _, image_2 = _render.render(channel_2.locs, channel_2.info)

    if channel_align_method == 'RCC':
        shift_y, shift_x = rcc([image_1, image_2])
        red_to_green_transform_mat = (0, -shift_y[1], -shift_x[1])
        aligned_channel_2_movie = shift(channel_2.movie, red_to_green_transform_mat, mode='constant', cval=0, order=0)

    elif channel_align_method == 'StackReg':
        sr = StackReg(StackReg.RIGID_BODY)
        red_to_green_transform_mat = sr.register(image_1, image_2)
        aligned_channel_2_movie = stackreg_channel_alignment(mov=channel_2.movie, transfer_matrix=red_to_green_transform_mat)

    # Note the types of red_to_green_transform_mat are different with two method

    else:
        raise ValueError('channel_align_method is either RCC or StackReg')

    aligned_ref = np.concatenate((channel_1.movie, aligned_channel_2_movie), axis=2)

    if save:
        imwrite(ref_movie_path.replace('.tif', '_corrected.tif'), aligned_ref)

    # use the first frame of the reference movie as the reference image
    # the fiducial markers will stand out in the green channel
    green_ref_image = channel_1.movie[0, :, :]

    # identify locs belonging to fiducial markers
    for movie_path in movie_path_list:
        green, red = prepare_two_channel_movie(movie_path, gpu=gpu, gradient_1=5000, gradient_2=400,
                                               box_1=5, box_2=5, roi=roi)
        green_image = green.movie[0, :, :]

        if channel_align_method == 'RCC':
            # align red to green
            red_aligned_movie = shift(red.movie, red_to_green_transform_mat, mode='constant', cval=0, order=0)

            # align green to green_ref
            shift_y, shift_x = rcc([green_ref_image, green_image])
            green_to_green_ref_mat = (0, -shift_y[1], -shift_x[1])
            green_aligned_movie = shift(green.movie, green_to_green_ref_mat, mode='constant', cval=0, order=0)

            # align red to green_ref
            red_aligned_movie = shift(red_aligned_movie, green_to_green_ref_mat, mode='constant', cval=0, order=0)

        else:
            # align red to green
            red_aligned_movie = stackreg_channel_alignment(mov=red.movie, transfer_matrix=red_to_green_transform_mat)

            # align green to green_ref
            sr = StackReg(StackReg.RIGID_BODY)
            green_to_green_ref_mat = sr.register(green_ref_image, green_image)
            green_aligned_movie = stackreg_channel_alignment(mov=green.movie, transfer_matrix=green_to_green_ref_mat)

            # align red to green_ref
            red_aligned_movie = stackreg_channel_alignment(mov=red_aligned_movie, transfer_matrix=green_to_green_ref_mat)

        # concat two channels
        aligned_movie = np.concatenate((green_aligned_movie, red_aligned_movie), axis=2)

        if save:
            imwrite(movie_path.replace('.tif', '_corrected.tif'), aligned_movie)


    return


import os
def file_sort(dir_path):
    file_list = os.listdir(dir_path)
    file_list = [x for x in file_list if x.endswith('.tif')]
    trans_mov = [x for x in file_list if 'ALEX' in x]

    if len(trans_mov) != 1:
        raise ValueError('There should be only one transcription movie in the folder')
    trans_mov = trans_mov[0]


    file_list.remove(trans_mov)

    trans_mov = dir_path + '/' + trans_mov
    file_list = [dir_path + '/' + x for x in file_list]


    return trans_mov, file_list


if __name__ == "__main__":

    # ref = "H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_trans_ALEX.tif"
    # #ref_roi = [0, 0, 684, 420]  # green channel # Note that two NIM have different width
    #
    # mov_list = ["H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_degen100nM_S4A5D300nM-2.tif",
    #         "H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_degen100nM_S4C5D300nM.tif",
    #         "H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_degen100nM_S4D5A300nM.tif",
    #         "H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_degen100nM_S4D5C300nM.tif",
    #         "H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_degen100nM_S4D5G300nM.tif",
    #         "H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_degen100nM_S4D5T300nM.tif",
    #         "H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_degen100nM_S4G5D300nM.tif",
    #         "H:/jagadish_data/transcription-GAPseq/IPE_trans_NTP200_degenAtto647Nexp7_IPE_degen100nM_S4T5D300nM.tif"]
    # #roi = [0, 428, 684, 856]  # red channel
    #
    # # position_correction_locs(mov_list, ref, roi=roi, ref_roi=None, save=False, gpu=True,
    # #                     channel_align_method='StackReg', ref_transcription=True)

    ref = "Y:/Qing_2/remote_interpreter_test/IPE_trans_NTP200Exp17_IPE_trans_ALEX.tif"
    mov_list = ["Y:/Qing_2/remote_interpreter_test/IPE_trans_NTP200Exp17_IPE_degen100nM_S4C5D300nM.tif",
                "Y:/Qing_2/remote_interpreter_test/IPE_trans_NTP200Exp17_IPE_degen100nM_S4A5D300nM.tif"]

    position_correction_fiducial(mov_list, ref, roi=None, ref_roi=None,
                                     gpu=True, save=True, channel_align_method='StackReg')






