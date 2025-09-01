import matplotlib.pyplot as plt
import numpy as np
import picasso.render as _render
from picasso.io import save_locs
from tifffile import imwrite, imread
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pystackreg import StackReg
from pystackreg.util import to_uint16
from picasso_utils import one_channel_movie
from DeepFRET_utils import subtract_background_deepFRET

try:
    import cupy as cp
except:
    pass


def channel_separate(movie_path):
    movie = imread(movie_path)
    w = movie.shape[2]

    channel_1 = one_channel_movie(movie[:, :, :w//2])
    channel_2 = one_channel_movie(movie[:, :, w//2:])

    channel_1.movie_format()
    channel_2.movie_format()

    return channel_1, channel_2


def prepare_two_channel_movie(movie_path, gradient_1=1000, drift_correction=True,
                              gradient_2=1000, box_1=5, box_2=5, gpu=True):

    channel_1, channel_2 = channel_separate(movie_path)

    if drift_correction:
        channel_1.lq_fitting(GPU=gpu, gradient=gradient_1, box=box_1)
        channel_2.lq_fitting(GPU=gpu, gradient=gradient_2, box=box_2)

        if len(channel_1.locs) > 0:
            channel_1.drift_correction(gpu)
        if len(channel_2.locs) > 0:
            channel_2.drift_correction(gpu)


    return channel_1, channel_2


def process_frame(frame, transform_mat, sr):
    """
    Process a single frame (CPU-bound task).
    """

    return sr.transform(frame, tmat=transform_mat)


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


def stackreg_channel_alignment(mov, transform_matrix, num_processes=4):
    """
    Align all frames in `mov` using hybrid parallelism.
    """

    sr = StackReg(StackReg.RIGID_BODY)

    # Split the frames into chunks for multiprocessing
    num_frames = mov.shape[0]
    num_processes = num_processes or multiprocessing.cpu_count()

    # Use multiprocessing to process chunks in parallel
    if num_frames < num_processes:
        # Process all frames in a single chunk without multiprocessing
        aligned_mov = np.array(process_frame_chunk(mov, transform_matrix, sr))
    else:
        # Calculate chunk_size only when num_frames >= num_processes
        chunk_size = max(1, num_frames // num_processes)  # Ensure chunk_size is at least 1
        frame_chunks = [mov[i:i + chunk_size] for i in range(0, num_frames, chunk_size)]

        # Use multiprocessing as before
        chunk_results = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            for r in pool.starmap(process_frame_chunk, [(chunk, transform_matrix, sr) for chunk in frame_chunks]):
                chunk_results.append(r)
        aligned_mov = np.concatenate(chunk_results, axis=0)


    return to_uint16(aligned_mov)




def contrast_enhance(img, gamma_high=0.2, gamma_low=5.0):
    # make fiducial markers more significant

    img_scaled = (img - img.min()) / (img.max() - img.min())

    # Dual gamma correction
    threshold = np.percentile(img, 98)  # point for gamma split
    bright_mask = img_scaled > threshold
    dark_mask = ~bright_mask

    img_gamma = np.zeros_like(img_scaled)
    img_gamma[bright_mask] = 65535 * np.power(img_scaled[bright_mask], gamma_high)
                                    #(img_scaled[bright_mask] / 255) ** (1 / gamma_high)
    img_gamma[dark_mask] = 65535 * np.power(img_scaled[dark_mask], gamma_low)
                                #255 * (img_scaled[dark_mask] / 255) ** gamma_low
    img_gamma = np.clip(img_gamma, 0, 65535).astype(np.uint16)

    plt.imshow(img_gamma, cmap='gray')
    plt.grid(None)
    plt.show()

    return img_gamma


def align_red_green(movie_path, alignment_source, background_remove, gpu):
    channel_1, channel_2 = prepare_two_channel_movie(movie_path, gpu=gpu, drift_correction=True)

    if alignment_source == 'super-resolution':
        _, image_1 = _render.render(channel_1.locs, channel_1.info)
        _, image_2 = _render.render(channel_2.locs, channel_2.info)

    elif alignment_source == 'first':
        image_1 = contrast_enhance(channel_1.movie[0, :, :])
        image_2 = contrast_enhance(channel_2.movie[0, :, :])

    else:
        raise ValueError('the image used to calculate red to green transformation matrix is either'
                         'constructed super-resolution images or the first frame from movies')

    sr = StackReg(StackReg.RIGID_BODY)
    red_to_green_transform_mat = sr.register(image_1, image_2)

    if background_remove:
        channel_2.movie = subtract_background_deepFRET(channel_2.movie)
        channel_1.movie = subtract_background_deepFRET(channel_1.movie)

    aligned_channel_2_movie = stackreg_channel_alignment(mov=channel_2.movie,
                                                        transform_matrix=red_to_green_transform_mat)


    aligned_ref = np.concatenate((channel_1.movie, aligned_channel_2_movie), axis=2)

    return aligned_ref, red_to_green_transform_mat





def two_step_channel_align(movie_path, green_ref_image, red_to_green_transform_mat,
                           alignment_source, gpu, background_remove=False):

    green, red = prepare_two_channel_movie(movie_path, gpu=gpu)
    if alignment_source == 'super-resolution':
        _, green_image = _render.render(green.locs, green.info)

    else:
        green_image = green.movie[0, :, :]

    # align green to green_ref
    sr = StackReg(StackReg.RIGID_BODY)
    green_to_green_ref_mat = sr.register(green_ref_image, green_image)

    if background_remove:
        green.movie = subtract_background_deepFRET(green.movie)
        red.movie = subtract_background_deepFRET(red.movie)

    #align green to green_ref
    green_aligned_movie = stackreg_channel_alignment(mov=green.movie, transform_matrix=green_to_green_ref_mat)
    # align red to green_ref
    red_aligned_movie = stackreg_channel_alignment(red.movie,
                                        transform_matrix=np.dot(green_to_green_ref_mat, red_to_green_transform_mat))

    # concat two channels
    aligned_movie = np.concatenate((green_aligned_movie, red_aligned_movie), axis=2)


    return aligned_movie



def position_correction_fiducial(movie_path_list, ref_movie_path, gpu=True,
                                 gg_alignment_source='first', rg_alignment_source='first',
                                 ref_background_remove=False, mov_background_remove=False):
    ''' This function do locs_based_analysis between different green channels using fiducial markers.
    And the fiducial markers can be detected easily in green channel but not red channel.
    The locs in green and red channels of transcription movie should be co-localized. Thus,
    the transformation matrix between green and red channel are acquired from there. '''
    aligned_ref, red_to_green_transform_mat = align_red_green(ref_movie_path,
                                                              alignment_source=rg_alignment_source,
                                                              background_remove=ref_background_remove,
                                                              gpu=gpu)
    print(red_to_green_transform_mat)
    imwrite(ref_movie_path.replace('.tif', '_corrected.tif'), aligned_ref)

    # use the first frame of the reference movie as the reference image
    # the fiducial markers will stand out in the green channel
    channel_1, channel_2 = channel_separate(ref_movie_path)
    if gg_alignment_source == 'super-resolution':
        channel_1.lq_fitting(gradient=1000, GPU=gpu)
        _, green_ref_image = _render.render(channel_1.locs, channel_1.info)
    elif gg_alignment_source == 'first':
        green_ref_image = channel_1.movie[0, :, :]
    else:
        raise ValueError('alignment_source must be either constructed super-resolution or first')

    for movie_path in movie_path_list:
        aligned_movie = two_step_channel_align(movie_path, green_ref_image,
                                               red_to_green_transform_mat,
                                               gg_alignment_source, gpu,
                                               mov_background_remove)
        imwrite(movie_path.replace('.tif', '_corrected.tif'), aligned_movie)


    return



import os

def process_correction(dir_path, localization_key='localization', rg_alignment_source='first',
                       gg_alignment_source='first', gpu=True,
                       ref_background_remove=False, mov_background_remove=False):
    files = [x for x in os.listdir(dir_path) if x.endswith('.tif') or x.endswith('.raw')]
    ref_list = [x for x in files if localization_key in x]

    if len(ref_list) == 1:
        ref_path =os.path.join(dir_path, ref_list[0])
        files.remove(ref_list[0])
        mov_path = [os.path.join(dir_path, x) for x in files]

    elif len(ref_list) > 1:
        # when multiple reference movies detected use the first one recorded
        mov_path = [x for x in files if x not in ref_list]
        mov_path = [os.path.join(dir_path, x) for x in mov_path]

        ref_path_list = [os.path.join(dir_path, x) for x in ref_list]
        ref_path = min(ref_path_list, key=os.path.getmtime)

        ref_path_list.remove(ref_path)
        mov_path.extend(ref_path_list)

    else:
        raise ValueError('no reference file is found')

    position_correction_fiducial(mov_path, ref_path, gpu=gpu,
                                 gg_alignment_source=gg_alignment_source,
                                 rg_alignment_source=rg_alignment_source,
                                 ref_background_remove=ref_background_remove,
                                 mov_background_remove=mov_background_remove)

    return


if __name__ == "__main__":
    process_correction("G:/CAP_dwellTime_analysis/20250708_CAP_C_2.5nM_100mMNaCl",
                       rg_alignment_source='first',
                       gg_alignment_source='first',
                       localization_key='localization', gpu=True,
                       ref_background_remove=False, mov_background_remove=False)

    # channel_1, channel_2 = prepare_two_channel_movie("G:/Miri_GapSeq/20240802_GapSeq_8mer_Tween/100nM/undrift/"
    #                                                  "GapSeq_8mer_Tween_GapA_100nM8merA643BhQ1_200ms_8gl_6r.tif",
    #                                                  gradient_1=500, gradient_2=500)
    # undrifted = np.concatenate((channel_1.movie, channel_2.movie), axis=2)
    # imwrite("G:/Miri_GapSeq/20240802_GapSeq_8mer_Tween/100nM/undrift/"
    #             "GapSeq_8mer_Tween_GapA_100nM8merA643BhQ1_200ms_8gl_6r_undrifted.tif", undrifted)
