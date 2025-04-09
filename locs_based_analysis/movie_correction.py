import matplotlib.pyplot as plt
import numpy as np
import picasso.render as _render
from tifffile import imwrite, imread
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pystackreg import StackReg
from pystackreg.util import to_uint16
from picasso_utils import one_channel_movie
import cupy as cp
from scipy.ndimage import median_filter


def channel_separate(movie_path):
    movie = imread(movie_path)
    h, w = movie.shape[1], movie.shape[2]

    rio_1 = [0, 0, h, w // 2]
    rio_2 = [0, w // 2, h, w]

    channel_1 = one_channel_movie(movie_path, roi=rio_1)
    channel_2 = one_channel_movie(movie_path, roi=rio_2)

    channel_1.movie_format()
    channel_2.movie_format()

    return channel_1, channel_2


def prepare_two_channel_movie(movie_path, gradient_1=1000, drift_correction=True,
                              gradient_2=1000, box_1=5, box_2=5, gpu=True):

    channel_1, channel_2 = channel_separate(movie_path)

    if drift_correction:
        channel_1.lq_fitting(GPU=gpu, min_net_gradient=gradient_1, box=box_1)
        channel_2.lq_fitting(GPU=gpu, min_net_gradient=gradient_2, box=box_2)

        channel_1.drift_correction(gpu)
        channel_2.drift_correction(gpu)


    return channel_1, channel_2



def process_frame(frame, transform_mat, sr):
    """
    Process a single frame (CPU-bound task).
    """

    return sr.transform(frame, tmat=transform_mat)


def process_frame_chunk(frame_chunk, transform_mat, sr, max_threads=4):
    """
    Process a chunk of frames using multithreading.
    """
    with ThreadPoolExecutor(max_workers=max_threads) as thread_pool:
        # Process all frames in the chunk in parallel using threads
        results = list(thread_pool.map(
            lambda frame: process_frame(frame, transform_mat, sr),
            frame_chunk
        ))
    return results


def stackreg_channel_alignment(mov, transfer_matrix, num_processes=4):
    """
    Align all frames in `mov` using hybrid parallelism.
    """

    sr = StackReg(StackReg.RIGID_BODY)
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

    return to_uint16(aligned_mov)



def contrast_enhance(image):
    if len(image.shape) == 3:
        image = image[0, :, :]

    bg = median_filter(image.astype(np.float32), size=21)
    image = image - bg
    image = np.where(image < 0, 0, image)

    min_img = np.min(image)
    max_img = np.max(image)
    image = (image - min_img) * (255.0 / (max_img - min_img))

    return image.astype(np.uint16)


def align_red_green(movie_path, alignment_source, gpu=True):
    channel_1, channel_2 = prepare_two_channel_movie(movie_path, gpu=gpu, drift_correction=True)
    if alignment_source == 'super-resolution':
        _, image_1 = _render.render(channel_1.locs, channel_1.info)
        _, image_2 = _render.render(channel_2.locs, channel_2.info)

    elif alignment_source == 'first':
        image_1 = contrast_enhance(channel_1.movie)
        image_2 = contrast_enhance(channel_2.movie)

    else:
        raise ValueError('the image used to calculate red to green transformation matrix is either'
                         'constructed super-resolution images or the first frame from movies')

    sr = StackReg(StackReg.RIGID_BODY)
    red_to_green_transform_mat = sr.register(image_1, image_2)

    aligned_channel_2_movie = stackreg_channel_alignment(mov=channel_2.movie,
                                                             transfer_matrix=red_to_green_transform_mat)

    aligned_ref = np.concatenate((channel_1.movie, aligned_channel_2_movie), axis=2)

    return aligned_ref, red_to_green_transform_mat





def two_step_channel_align(movie_path, green_ref_image, red_to_green_transform_mat, gpu):

    green, red = prepare_two_channel_movie(movie_path, gpu=gpu, gradient_1=5000, gradient_2=1000,
                                           box_1=5, box_2=5, drift_correction=True)
    green_image = green.movie[0, :, :]
    #green_image = contrast_enhance(green_image)


    # align red to green
    red_aligned_movie = stackreg_channel_alignment(mov=red.movie, transfer_matrix=red_to_green_transform_mat)

    # align green to green_ref
    sr = StackReg(StackReg.RIGID_BODY)
    green_to_green_ref_mat = sr.register(green_ref_image, green_image)
    green_aligned_movie = stackreg_channel_alignment(mov=green.movie, transfer_matrix=green_to_green_ref_mat)

    # align red to green_ref
    red_aligned_movie = stackreg_channel_alignment(mov=red_aligned_movie, transfer_matrix=green_to_green_ref_mat)
    # if bg_remove is not None:
    #     red_aligned_movie = background_remove(red_aligned_movie, method=bg_remove, gpu=gpu)

    # concat two channels
    aligned_movie = np.concatenate((green_aligned_movie, red_aligned_movie), axis=2)


    return aligned_movie



def position_correction_fiducial(movie_path_list, ref_movie_path, gpu=True,
                                 alignment_source='super-resolution'):
    ''' This function do locs_based_analysis between different green channels using fiducial markers.
    And the fiducial markers can be detected easily in green channel but not red channel.
    The locs in green and red channels of transcription movie should be co-localized. Thus,
    the transformation matrix between green and red channel are acquired from there. '''

    aligned_ref, red_to_green_transform_mat = align_red_green(ref_movie_path, alignment_source=alignment_source,
                                                              gpu=gpu)
    print(red_to_green_transform_mat)
    imwrite(ref_movie_path.replace('.tif', '_corrected.tif'), aligned_ref)

    # use the first frame of the reference movie as the reference image
    # the fiducial markers will stand out in the green channel
    channel_1, channel_2 = channel_separate(ref_movie_path)
    green_ref_image = channel_1.movie[0, :, :]
    #green_ref_image = contrast_enhance(green_ref_image)


    for movie_path in movie_path_list:
        aligned_movie = two_step_channel_align(movie_path, green_ref_image, red_to_green_transform_mat, gpu)
        imwrite(movie_path.replace('.tif', '_corrected.tif'), aligned_movie)


    return



import os

def process_correction_Localization(dir_path, localization_key='localization'):
    files = [x for x in os.listdir(dir_path) if x.endswith('.tif')]
    ref = [x for x in files if localization_key in x]
    if len(ref) != 1:
        raise ValueError("There should be one and only one reference file in the directory")

    files.remove(ref[0])
    ref_movie_path = os.path.join(dir_path, ref[0])

    movie_path_list = [os.path.join(dir_path, x) for x in files]

    position_correction_fiducial(movie_path_list, ref_movie_path, gpu=True, alignment_source='first')

    return



def process_correction_photobleaching(dir_path, gpu=True):
    files = [x for x in os.listdir(dir_path) if x.endswith('.tif')]
    localization_path = [x for x in files if 'Localization' in x or 'localization' in x]

    mov_path = [x for x in files if x not in localization_path]

    localization_path = [os.path.join(dir_path, x) for x in localization_path]
    mov_path = [os.path.join(dir_path, x) for x in mov_path]

    # -------------------get red to green transformation matrix------------------
    # the first movie (no phot0-bleaching) should have the strongest signal
    ref_path = min(localization_path, key=os.path.getmtime)

    # align the result of localization movie and gap movies in thee same way
    localization_path.remove(ref_path)
    mov_path.extend(localization_path)


    position_correction_fiducial(mov_path, ref_path, alignment_source='first',gpu=gpu)

    return



if __name__ == "__main__":
    process_correction_Localization("G:/20250405_IPE_NTP200_ALEX_exp29/original_files",
                                    localization_key='combined')
    #process_correction_photobleaching('H:/photobleaching/20250328_8nt_GAP_photobleach2')

