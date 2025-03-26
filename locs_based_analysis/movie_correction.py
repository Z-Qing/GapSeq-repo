import matplotlib.pyplot as plt
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


def prepare_two_channel_movie(movie_path, gradient_1=400, get_locs=False,
                              gradient_2=400, box_1=5, box_2=5, gpu=True, bg_filtering=None):
    movie = imread(movie_path)
    h, w = movie.shape[1], movie.shape[2]

    rio_1 = [0, 0, h, w//2]
    rio_2 = [0, w//2, h, w]

    channel_1 = one_channel_movie(movie_path, roi=rio_1)
    channel_2 = one_channel_movie(movie_path, roi=rio_2)

    if bg_filtering == 'gaussian':
        channel_1.gaussian_filter()
        channel_2.gaussian_filter()
    elif bg_filtering == 'median':
        channel_1.median_filter()
        channel_2.median_fulter()
    else:
        #raise Warning('un-supported background filtering method')
        pass


    if get_locs:
        channel_1.lq_fitting(GPU=gpu, min_net_gradient=gradient_1, box=box_1)
        channel_2.lq_fitting(GPU=gpu, min_net_gradient=gradient_2, box=box_2)

        channel_1.drift_correction(gpu)
        channel_2.drift_correction(gpu)

    else:
        channel_1.io_movie_format()
        channel_2.io_movie_format()

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

    # Use multiprocessing to process chunks in parallel
    if num_frames < num_processes:
        # Process all frames in a single chunk without multiprocessing
        aligned_mov = np.array(process_frame_chunk(mov, transform_mat, sr))
    else:
        # Calculate chunk_size only when num_frames >= num_processes
        chunk_size = max(1, num_frames // num_processes)  # Ensure chunk_size is at least 1
        frame_chunks = [mov[i:i + chunk_size] for i in range(0, num_frames, chunk_size)]

        # Use multiprocessing as before
        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_results = pool.starmap(
                process_frame_chunk,
                [(chunk, transform_mat, sr) for chunk in frame_chunks]
            )
        aligned_mov = np.concatenate(chunk_results, axis=0)

    return aligned_mov



def position_correction_fiducial(movie_path_list, ref_movie_path, gpu=True, alignment_method='RCC',
                                 alignment_source='super-resolution'):
    ''' This function do locs_based_analysis between different green channels using fiducial markers.
    And the fiducial markers can be detected easily in green channel but not red channel.
    The locs in green and red channels of transcription movie should be co-localized. Thus,
    the transformation matrix between green and red channel are acquired from there. '''

    # create image for channel locs_based_analysis
    if alignment_source == 'super-resolution':
        channel_1, channel_2 = prepare_two_channel_movie(ref_movie_path, gpu=gpu, get_locs=True)
        _, image_1 = _render.render(channel_1.locs, channel_1.info)
        _, image_2 = _render.render(channel_2.locs, channel_2.info)

    elif alignment_source == 'first':
        channel_1, channel_2 = prepare_two_channel_movie(ref_movie_path, gpu=gpu, get_locs=False)
        image_1 = channel_1.movie[0, :, :]
        image_2 = channel_2.movie[0, :, :]

        image_1 = (image_1 - np.min(image_1)) * (255.0 / (np.max(image_1) - np.min(image_1)))
        image_2 = (image_2 - np.min(image_2)) * (255.0 / (np.max(image_2) - np.min(image_2)))

        gamma = 1.5
        image_1 = (np.power(image_1 / 255.0, gamma) * 255.0).astype(np.uint16)
        image_2 = (np.power(image_2 / 255.0, gamma) * 255.0).astype(np.uint16)


    else:
        raise ValueError('the image used to calculate red to green transformation matrix is either'
                         'constructed super-resolution images or the first frame from movies')

    if alignment_method == 'RCC':
        shift_y, shift_x = rcc([image_1, image_2])
        red_to_green_transform_mat = (0, -shift_y[1], -shift_x[1])
        aligned_channel_2_movie = shift(channel_2.movie, red_to_green_transform_mat, mode='constant', cval=0, order=0)

    elif alignment_method == 'StackReg':
        sr = StackReg(StackReg.RIGID_BODY)
        red_to_green_transform_mat = sr.register(image_1, image_2)
        print(red_to_green_transform_mat)
        aligned_channel_2_movie = stackreg_channel_alignment(mov=channel_2.movie, transfer_matrix=red_to_green_transform_mat)

    # Note the types of red_to_green_transform_mat are different with two method
    else:
        raise ValueError('channel_align_method is either RCC or StackReg')

    aligned_ref = np.concatenate((channel_1.movie, aligned_channel_2_movie), axis=2)

    imwrite(ref_movie_path.replace('.tif', '_corrected.tif'), aligned_ref)

    # use the first frame of the reference movie as the reference image
    # the fiducial markers will stand out in the green channel
    green_ref_image = channel_1.movie[0, :, :]

    # identify locs belonging to fiducial markers
    for movie_path in movie_path_list:
        green, red = prepare_two_channel_movie(movie_path, gpu=gpu, gradient_1=5000, gradient_2=400,
                                               box_1=5, box_2=5, get_locs=True)
        green_image = green.movie[0, :, :]

        if alignment_method == 'RCC':
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

        imwrite(movie_path.replace('.tif', '_corrected.tif'), aligned_movie)

    return



import os

def process_correction_Localization(dir_path):
    files = [x for x in os.listdir(dir_path) if x.endswith('.tif')]
    ref = [x for x in files if 'Localization' in x or 'localization' in x]
    if len(ref) != 1:
        raise ValueError("There should be one and only one reference file in the directory")

    files.remove(ref[0])
    ref = os.path.join(dir_path, ref[0])

    mov_list = [os.path.join(dir_path, x) for x in files]

    position_correction_fiducial(mov_list, ref, alignment_source='first', gpu=True, alignment_method='StackReg')


    return


def process_correction_ALEX(dir_path):
    file_list = os.listdir(dir_path)
    file_list = [x for x in file_list if x.endswith('.tif')]
    trans_mov = [x for x in file_list if 'ALEX' in x]

    if len(trans_mov) != 1:
        raise ValueError('There should be only one transcription movie in the folder')

    trans_mov = trans_mov[0]
    file_list.remove(trans_mov)

    trans_mov = dir_path + '/' + trans_mov
    file_list = [dir_path + '/' + x for x in file_list]

    position_correction_fiducial(file_list, trans_mov, gpu=True, alignment_method='StackReg',
                                 alignment_source='super-resolution')

    return


if __name__ == "__main__":
    # ref_roi = [0, 0, 684, 420]  # green channel # Note that two NIM have different width
    # roi = [0, 428, 684, 856]  # red channel

    process_correction_Localization("H:/competitive/20250325_8nt_comp_GAP_C")







