import matplotlib.pyplot as plt
import numpy as np
import picasso.render as _render
from picasso.clusterer import dbscan
import cv2
from skimage.registration import phase_cross_correlation
from tifffile import imwrite, imread
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pystackreg import StackReg
from pystackreg.util import to_uint16
from picasso_utils import one_channel_movie
import os
from scipy.ndimage import binary_dilation, label


try:
    import cupy as cp
except:
    pass


def channel_separate(movie_path):
    movie = imread(movie_path)
    w = movie.shape[2]

    channel_1 = one_channel_movie(movie[:, :, :w//2])
    channel_2 = one_channel_movie(movie[:, :, w//2:])

    return channel_1, channel_2


def prepare_two_channel_movie(movie_path, gradient_1=1000, drift_correction=True,
                              gradient_2=1000, box_1=5, box_2=5, gpu=True):

    channel_1, channel_2 = channel_separate(movie_path)
    channel_1.movie_format()
    channel_2.movie_format()

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


def img_rescale(img, gamma_high=0.5, gamma_low=2, percentile=90,
                     gamma_correction=True):
    lo = img.min()
    hi = img.max()
    if hi <= lo: hi = lo + 1e-6
    z = (img - lo) / (hi - lo)
    # gamma < 1 brightens highlights
    if gamma_correction:
        gamma_z = np.power(z, gamma_high, where=z > np.percentile(z, percentile),
                     out=np.power(z, gamma_low))
        z = gamma_z

    z = np.clip(z, 0, 1)
    z = z * 255

    return z.astype(np.uint8)



def contrast_enhance_fiducial(img, threshold=2000, box=7,
                           out_range=(0, 255), percentiles=(1, 99), # robust min/max; set to (0,100) for true min/max
                           show=True, dtype=np.uint8, connectivity=8):

    I = np.asarray(img)
    if I.ndim != 2:
        raise ValueError("img must be a 2D array")

    # Ensure odd box
    if box < 1:
        raise ValueError("box must be >= 1")
    if box % 2 == 0:
        box += 1

    # Picked pixels via threshold + dilation
    seeds = I >= float(threshold)
    footprint = np.ones((box, box), dtype=bool)
    keep_mask = binary_dilation(seeds, structure=footprint)

    vals = I.astype(np.float32, copy=False)
    out_lo, out_hi = map(float, out_range)
    out = np.zeros_like(vals, dtype=np.float32)

    if not np.any(keep_mask):
        # nothing picked
        if show:
            plt.imshow(out, cmap='gray'); plt.axis('off'); plt.grid(False); plt.show()
        return out.astype(dtype) if dtype is not None else out


    # label components (boxes)
    structure = np.ones((3,3), bool) if connectivity == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], bool)
    labels, n = label(keep_mask, structure=structure)

    # Compute robust per-label lows/highs via bincount on ranks
    # Simpler: loop (fast enough for typical #boxes)
    for k in range(1, n+1):
        m = labels == k
        pv = vals[m]
        lo, hi = np.percentile(pv, percentiles)
        den = max(hi - lo, 1e-12)
        s = (vals[m] - lo) / den
        s = np.clip(s, 0.0, 1.0)
        out[m] = s * (out_hi - out_lo) + out_lo

    # Cast if requested
    if dtype is not None:
        if np.issubdtype(np.dtype(dtype), np.integer):
            out = np.clip(out, out_lo, out_hi)
        out = out.astype(dtype)

    if show:
        plt.imshow(out, cmap='gray'); plt.axis('off'); plt.grid(False); plt.show()

    return out



def align_red_green(movie_path,  gpu):
    channel_1, channel_2 = prepare_two_channel_movie(movie_path, gpu=gpu, drift_correction=True)

    image_1 = contrast_enhance_fiducial(channel_1.movie[0, :, :], threshold=50000)
    image_2 = contrast_enhance_fiducial(channel_2.movie[0, :, :], threshold=1000)

    sr = StackReg(StackReg.RIGID_BODY)

    rg_transform_mat = sr.register(image_2, image_1)
    aligned_channel_1_movie = stackreg_channel_alignment(mov=channel_1.movie, transform_matrix=rg_transform_mat)
    aligned_ref = np.concatenate((aligned_channel_1_movie, channel_2.movie), axis=2)

    return aligned_ref, rg_transform_mat, image_2


def two_step_channel_align(movie_path, ref_image, rg_transform_mat, gpu, alignment_source, method):
    # align green to green_ref
    green, red = prepare_two_channel_movie(movie_path, gpu=gpu)

    if alignment_source == 'super-resolution':
        red_cluster = dbscan(red.locs, radius=1, min_samples=50)
        _, red_image = _render.render(red_cluster, red.info)
        red_image = img_rescale(red_image)
        if method == 'phase':
            shift, _, _ = phase_cross_correlation(ref_image, red_image,
                                                  overlap_ratio=0.95, upsample_factor=50)
            red_to_red_ref = np.array([[1, 0, -shift[1]], [0, 1, -shift[0]], [0, 0, 1]]).astype(np.float32)

        elif method == 'real':
            sr = StackReg(StackReg.RIGID_BODY)
            red_to_red_ref = sr.register(ref_image, red_image)
        else:
            raise ValueError("method must be 'phase' or 'real'")

    elif alignment_source == 'fiducial':
        red_image = red.movie[0, :, :]
        red_image = contrast_enhance_fiducial(red_image)
        sr = StackReg(StackReg.RIGID_BODY)
        red_to_red_ref = sr.register(ref_image, red_image)

    else:
        raise ValueError('alignment_source must be either "super-resolution" or "fiducial"')


    print(red_to_red_ref)
    red_aligned = stackreg_channel_alignment(mov=red.movie, transform_matrix= red_to_red_ref)
    green_aligned = stackreg_channel_alignment(mov=green.movie,
                                                   transform_matrix=np.dot(red_to_red_ref, rg_transform_mat))
    aligned_movie = np.concatenate((green_aligned, red_aligned), axis=2)

    return aligned_movie



def position_correction(movie_path_list, locs_movie_path, gpu=True, method='phase'):
    ''' This function do locs_based_analysis between different green channels using fiducial markers.
    And the fiducial markers can be detected easily in green channel but not red channel.
    The locs in green and red channels of transcription movie should be co-localized. Thus,
    the transformation matrix between green and red channel are acquired from there. '''

    # align localization movie using fiducial markers
    aligned_locs_movie, rg_transform_mat, red_locs_ref = align_red_green(locs_movie_path, gpu=gpu)
    print(rg_transform_mat)
    imwrite(locs_movie_path.replace('.tif', '_corrected.tif'), aligned_locs_movie)

    # find the first movie after localization movie and align using fiducial markers
    # because the localization movie is only 10 frame can't create super-resolution image
    # from it
    fist_movie_path = min(movie_path_list, key=os.path.getmtime)
    aligned_first_mov = two_step_channel_align(fist_movie_path, red_locs_ref, rg_transform_mat,
                                               alignment_source='fiducial', gpu=gpu, method='real')
    imwrite(fist_movie_path.replace('.tif', '_corrected.tif'), aligned_first_mov)

    first_red = one_channel_movie(aligned_first_mov[:, :, 428:])
    first_red.movie_format()
    first_red.lq_fitting(GPU=gpu, gradient=1000)
    first_red_cluster = dbscan(first_red.locs, radius=1, min_samples=50)

    _, red_ref = _render.render(first_red_cluster, first_red.info)
    red_ref = img_rescale(red_ref)

    # for the rest of movies creating super-resolution images in red channel for registration
    movie_path_list.remove(fist_movie_path)
    for movie_path in movie_path_list:
        aligned_movie = two_step_channel_align(movie_path, red_ref, rg_transform_mat, method=method,
                                               alignment_source='super-resolution', gpu=gpu)
        imwrite(movie_path.replace('.tif', '_corrected.tif'), aligned_movie)

    return


def process_correction(dir_path, localization_key='localization', gpu=True, method='phase'):
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

    position_correction(mov_path, ref_path, gpu=gpu, method=method)

    return


if __name__ == "__main__":
    process_correction("G:/original",
                       localization_key='localization', gpu=True, method='phase')