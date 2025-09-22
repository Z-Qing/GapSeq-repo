"""
Implementation of the masked normalized cross-correlation.

Based on the following publication:
D. Padfield. Masked object registration in the Fourier domain.
IEEE Transactions on Image Processing (2012)

and the author's original MATLAB implementation, available on this website:
http://www.dirkpadfield.com/
"""
import itertools
import warnings
from scipy.fft import fftn, ifftn, fftfreq
from scipy import ndimage as ndi
from functools import partial
import numpy as np
import scipy.fft as fftmodule
from scipy.fft import next_fast_len

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,  # np.float128 ; doesn't exist on windows
    'G': np.complex128,  # np.complex256 ; doesn't exist on windows
}


#from .._shared.utils import _supported_float_type
def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or tuple of np.dtype
        The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def _masked_phase_cross_correlation(
    reference_image, moving_image, reference_mask, moving_mask=None, overlap_ratio=0.3
):
    """Masked image translation registration by masked normalized
    cross-correlation.

    Parameters
    ----------
    reference_image : ndarray
        Reference image.
    moving_image : ndarray
        Image to register. Must be same dimensionality as ``reference_image``,
        but not necessarily the same size.
    reference_mask : ndarray
        Boolean mask for ``reference_image``. The mask should evaluate
        to ``True`` (or 1) on valid pixels. ``reference_mask`` should
        have the same shape as ``reference_image``.
    moving_mask : ndarray or None, optional
        Boolean mask for ``moving_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels. ``moving_mask`` should have the same shape
        as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with numpy.

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`

    """
    if moving_mask is None:
        if reference_image.shape != moving_image.shape:
            raise ValueError(
                "Input images have different shapes, moving_mask must "
                "be explicitly set."
            )
        moving_mask = reference_mask.astype(bool)

    # We need masks to be of the same size as their respective images
    for im, mask in [(reference_image, reference_mask), (moving_image, moving_mask)]:
        if im.shape != mask.shape:
            raise ValueError("Image sizes must match their respective mask sizes.")

    xcorr = cross_correlate_masked(
        moving_image,
        reference_image,
        moving_mask,
        reference_mask,
        axes=tuple(range(moving_image.ndim)),
        mode='full',
        overlap_ratio=overlap_ratio,
    )

    # Generalize to the average of multiple equal maxima
    maxima = np.stack(np.nonzero(xcorr == xcorr.max()), axis=1)
    center = np.mean(maxima, axis=0)
    shifts = center - np.array(reference_image.shape) + 1

    # The mismatch in size will impact the center location of the
    # cross-correlation
    size_mismatch = np.array(moving_image.shape) - np.array(reference_image.shape)

    return -shifts + (size_mismatch / 2)


def cross_correlate_masked(
    arr1, arr2, m1, m2, mode='full', axes=(-2, -1), overlap_ratio=0.3
):
    """
    Masked normalized cross-correlation between arrays.

    Parameters
    ----------
    arr1 : ndarray
        First array.
    arr2 : ndarray
        Seconds array. The dimensions of `arr2` along axes that are not
        transformed should be equal to that of `arr1`.
    m1 : ndarray
        Mask of `arr1`. The mask should evaluate to `True`
        (or 1) on valid pixels. `m1` should have the same shape as `arr1`.
    m2 : ndarray
        Mask of `arr2`. The mask should evaluate to `True`
        (or 1) on valid pixels. `m2` should have the same shape as `arr2`.
    mode : {'full', 'same'}, optional
        'full':
            This returns the convolution at each point of overlap. At
            the end-points of the convolution, the signals do not overlap
            completely, and boundary effects may be seen.
        'same':
            The output is the same size as `arr1`, centered with respect
            to the `‘full’` output. Boundary effects are less prominent.
    axes : tuple of ints, optional
        Axes along which to compute the cross-correlation.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images.

    Returns
    -------
    out : ndarray
        Masked normalized cross-correlation.

    Raises
    ------
    ValueError : if correlation `mode` is not valid, or array dimensions along
        non-transformation axes are not equal.

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
    """
    if mode not in {'full', 'same'}:
        raise ValueError(f"Correlation mode '{mode}' is not valid.")

    fixed_image = np.asarray(arr1)
    moving_image = np.asarray(arr2)
    float_dtype = _supported_float_type((fixed_image.dtype, moving_image.dtype))
    if float_dtype.kind == 'c':
        raise ValueError("complex-valued arr1, arr2 are not supported")

    fixed_image = fixed_image.astype(float_dtype)
    fixed_mask = np.array(m1, dtype=bool)
    moving_image = moving_image.astype(float_dtype)
    moving_mask = np.array(m2, dtype=bool)
    eps = np.finfo(float_dtype).eps

    # Array dimensions along non-transformation axes should be equal.
    all_axes = set(range(fixed_image.ndim))
    for axis in all_axes - set(axes):
        if fixed_image.shape[axis] != moving_image.shape[axis]:
            raise ValueError(
                f'Array shapes along non-transformation axes should be '
                f'equal, but dimensions along axis {axis} are not.'
            )

    # Determine final size along transformation axes
    # Note that it might be faster to compute Fourier transform in a slightly
    # larger shape (`fast_shape`). Then, after all fourier transforms are done,
    # we slice back to`final_shape` using `final_slice`.
    final_shape = list(arr1.shape)
    for axis in axes:
        final_shape[axis] = fixed_image.shape[axis] + moving_image.shape[axis] - 1
    final_shape = tuple(final_shape)
    final_slice = tuple([slice(0, int(sz)) for sz in final_shape])

    # Extent transform axes to the next fast length (i.e. multiple of 3, 5, or
    # 7)
    fast_shape = tuple([next_fast_len(final_shape[ax]) for ax in axes])

    # We use the new scipy.fft because they allow leaving the transform axes
    # unchanged which was not possible with scipy.fftpack's
    # fftn/ifftn in older versions of SciPy.
    # E.g. arr shape (2, 3, 7), transform along axes (0, 1) with shape (4, 4)
    # results in arr_fft shape (4, 4, 7)
    fft = partial(fftmodule.fftn, s=fast_shape, axes=axes)
    _ifft = partial(fftmodule.ifftn, s=fast_shape, axes=axes)

    def ifft(x):
        return _ifft(x).real

    fixed_image[np.logical_not(fixed_mask)] = 0.0
    moving_image[np.logical_not(moving_mask)] = 0.0

    # N-dimensional analog to rotation by 180deg is flip over all relevant axes.
    # See [1] for discussion.
    rotated_moving_image = _flip(moving_image, axes=axes)
    rotated_moving_mask = _flip(moving_mask, axes=axes)

    fixed_fft = fft(fixed_image)
    rotated_moving_fft = fft(rotated_moving_image)
    fixed_mask_fft = fft(fixed_mask.astype(float_dtype))
    rotated_moving_mask_fft = fft(rotated_moving_mask.astype(float_dtype))

    # Calculate overlap of masks at every point in the convolution.
    # Locations with high overlap should not be taken into account.
    number_overlap_masked_px = ifft(rotated_moving_mask_fft * fixed_mask_fft)
    number_overlap_masked_px[:] = np.round(number_overlap_masked_px)
    number_overlap_masked_px[:] = np.fmax(number_overlap_masked_px, eps)
    masked_correlated_fixed_fft = ifft(rotated_moving_mask_fft * fixed_fft)
    masked_correlated_rotated_moving_fft = ifft(fixed_mask_fft * rotated_moving_fft)

    numerator = ifft(rotated_moving_fft * fixed_fft)
    numerator -= (
        masked_correlated_fixed_fft
        * masked_correlated_rotated_moving_fft
        / number_overlap_masked_px
    )

    fixed_squared_fft = fft(np.square(fixed_image))
    fixed_denom = ifft(rotated_moving_mask_fft * fixed_squared_fft)
    fixed_denom -= np.square(masked_correlated_fixed_fft) / number_overlap_masked_px
    fixed_denom[:] = np.fmax(fixed_denom, 0.0)

    rotated_moving_squared_fft = fft(np.square(rotated_moving_image))
    moving_denom = ifft(fixed_mask_fft * rotated_moving_squared_fft)
    moving_denom -= (
        np.square(masked_correlated_rotated_moving_fft) / number_overlap_masked_px
    )
    moving_denom[:] = np.fmax(moving_denom, 0.0)

    denom = np.sqrt(fixed_denom * moving_denom)

    # Slice back to expected convolution shape.
    numerator = numerator[final_slice]
    denom = denom[final_slice]
    number_overlap_masked_px = number_overlap_masked_px[final_slice]

    if mode == 'same':
        _centering = partial(_centered, newshape=fixed_image.shape, axes=axes)
        denom = _centering(denom)
        numerator = _centering(numerator)
        number_overlap_masked_px = _centering(number_overlap_masked_px)

    # Pixels where `denom` is very small will introduce large
    # numbers after division. To get around this problem,
    # we zero-out problematic pixels.
    tol = 1e3 * eps * np.max(np.abs(denom), axis=axes, keepdims=True)
    nonzero_indices = denom > tol

    # explicitly set out dtype for compatibility with SciPy < 1.4, where
    # fftmodule will be numpy.fft which always uses float64 dtype.
    out = np.zeros_like(denom, dtype=float_dtype)
    out[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
    np.clip(out, a_min=-1, a_max=1, out=out)

    # Apply overlap ratio threshold
    number_px_threshold = overlap_ratio * np.max(
        number_overlap_masked_px, axis=axes, keepdims=True
    )
    out[number_overlap_masked_px < number_px_threshold] = 0.0

    return out


def _centered(arr, newshape, axes):
    """Return the center `newshape` portion of `arr`, leaving axes not
    in `axes` untouched."""
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)

    slices = [slice(None, None)] * arr.ndim

    for ax in axes:
        startind = (currshape[ax] - newshape[ax]) // 2
        endind = startind + newshape[ax]
        slices[ax] = slice(startind, endind)

    return arr[tuple(slices)]


def _flip(arr, axes=None):
    """Reverse array over many axes. Generalization of arr[::-1] for many
    dimensions. If `axes` is `None`, flip along all axes."""
    if axes is None:
        reverse = [slice(None, None, -1)] * arr.ndim
    else:
        reverse = [slice(None, None, None)] * arr.ndim
        for axis in axes:
            reverse[axis] = slice(None, None, -1)

    return arr[tuple(reverse)]

"""
Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
"""



def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "shape of upsampled region sizes must be equal "
                "to input data's number of dimensions."
            )

    if axis_offsets is None:
        axis_offsets = [
            0,
        ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "number of axis offsets must be equal to input "
                "data's number of dimensions."
            )

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * fftfreq(
            n_items, upsample_factor
        )
        kernel = np.exp(-im2pi * kernel)
        # use kernel with same precision as the data
        kernel = kernel.astype(data.dtype, copy=False)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be
        zero if images are non-negative).

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image
    target_amp : float
        The normalized average image intensity of the target image
    """
    amp = src_amp * target_amp
    if amp == 0:
        warnings.warn(
            "Could not determine RMS error between images with the normalized "
            f"average intensities {src_amp!r} and {target_amp!r}. Either the "
            "reference or moving image may be empty.",
            UserWarning,
            stacklevel=3,
        )
    with np.errstate(invalid="ignore"):
        error = 1.0 - cross_correlation_max * cross_correlation_max.conj() / amp
    return np.sqrt(np.abs(error))


def _disambiguate_shift(reference_image, moving_image, shift):
    """Determine the correct real-space shift based on periodic shift.

    When determining a translation shift from phase cross-correlation in
    Fourier space, the shift is only correct to within a period of the image
    size along each axis, resulting in $2^n$ possible shifts, where $n$ is the
    number of dimensions of the image. This function checks the
    cross-correlation in real space for each of those shifts, and returns the
    one with the highest cross-correlation.

    The strategy we use is to perform the shift on the moving image *using the
    'grid-wrap' mode* in `scipy.ndimage`. The moving image's original borders
    then define $2^n$ quadrants, which we cross-correlate with the reference
    image in turn using slicing. The entire operation is thus $O(2^n + m)$,
    where $m$ is the number of pixels in the image (and typically dominates).

    Parameters
    ----------
    reference_image : numpy array
        The reference (non-moving) image.
    moving_image : numpy array
        The moving image: applying the shift to this image overlays it on the
        reference image. Must be the same shape as the reference image.
    shift : ndarray
        The shift to apply to each axis of the moving image, *modulo* image
        size. The length of ``shift`` must be equal to ``moving_image.ndim``.

    Returns
    -------
    real_shift : ndarray
        The shift disambiguated in real space.
    """
    shape = reference_image.shape
    positive_shift = [shift_i % s for shift_i, s in zip(shift, shape)]
    negative_shift = [shift_i - s for shift_i, s in zip(positive_shift, shape)]
    subpixel = np.any(np.array(shift) % 1 != 0)
    interp_order = 3 if subpixel else 0
    shifted = ndi.shift(moving_image, shift, mode='grid-wrap', order=interp_order)
    indices = np.round(positive_shift).astype(int)
    splits_per_dim = [(slice(0, i), slice(i, None)) for i in indices]
    max_corr = -1.0
    max_slice = None
    for test_slice in itertools.product(*splits_per_dim):
        reference_tile = np.reshape(reference_image[test_slice], -1)
        moving_tile = np.reshape(shifted[test_slice], -1)
        corr = -1.0
        if reference_tile.size > 2:
            # In the case of zero std, np.corrcoef returns NaN and warns
            # about division by zero. This is expected and handled below.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                corr = np.corrcoef(reference_tile, moving_tile)[0, 1]
        if corr > max_corr:
            max_corr = corr
            max_slice = test_slice
    if max_slice is None:
        warnings.warn(
            f"Could not determine real-space shift for periodic shift {shift!r} "
            f"as requested by `disambiguate=True` (disambiguation is degenerate).",
            stacklevel=3,
        )
        return shift
    real_shift_acc = []
    for sl, pos_shift, neg_shift in zip(max_slice, positive_shift, negative_shift):
        real_shift_acc.append(pos_shift if sl.stop is None else neg_shift)

    return np.array(real_shift_acc)


def phase_cross_correlation(
    reference_image,
    moving_image,
    *,
    upsample_factor=1,
    space="real",
    disambiguate=False,
    reference_mask=None,
    moving_mask=None,
    overlap_ratio=0.3,
    normalization="phase",
):
    """Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT [1]_.

    Parameters
    ----------
    reference_image : array
        Reference image.
    moving_image : array
        Image to register. Must be same dimensionality as
        ``reference_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).
        Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data. "real" means
        data will be FFT'd to compute the correlation, while "fourier"
        data will bypass FFT of input data. Case insensitive. Not
        used if any of ``reference_mask`` or ``moving_mask`` is not
        None.
    disambiguate : bool
        The shift returned by this function is only accurate *modulo* the
        image shape, due to the periodic nature of the Fourier transform. If
        this parameter is set to ``True``, the *real* space cross-correlation
        is computed for each possible shift, and the shift with the highest
        cross-correlation within the overlapping area is returned.
    reference_mask : ndarray
        Boolean mask for ``reference_image``. The mask should evaluate
        to ``True`` (or 1) on valid pixels. ``reference_mask`` should
        have the same shape as ``reference_image``.
    moving_mask : ndarray or None, optional
        Boolean mask for ``moving_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels. ``moving_mask`` should have the same shape
        as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images. Used only if one of ``reference_mask`` or
        ``moving_mask`` is not None.
    normalization : {"phase", None}
        The type of normalization to apply to the cross-correlation. This
        parameter is unused when masks (`reference_mask` and `moving_mask`) are
        supplied.

    Returns
    -------
    shift : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        the axis order of the input array.
    error : float
        Translation invariant normalized RMS error between
        ``reference_image`` and ``moving_image``. For masked cross-correlation
        this error is not available and NaN is returned.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative). For masked cross-correlation
        this phase difference is not available and NaN is returned.

    Notes
    -----
    The use of cross-correlation to estimate image translation has a long
    history dating back to at least [2]_. The "phase correlation"
    method (selected by ``normalization="phase"``) was first proposed in [3]_.
    Publications [1]_ and [2]_ use an unnormalized cross-correlation
    (``normalization=None``). Which form of normalization is better is
    application-dependent. For example, the phase correlation method works
    well in registering images under different illumination, but is not very
    robust to noise. In a high noise scenario, the unnormalized method may be
    preferable.

    When masks are provided, a masked normalized cross-correlation algorithm is
    used [5]_, [6]_.

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] P. Anuta, Spatial registration of multispectral and multitemporal
           digital imagery using fast Fourier transform techniques, IEEE Trans.
           Geosci. Electron., vol. 8, no. 4, pp. 353–368, Oct. 1970.
           :DOI:`10.1109/TGE.1970.271435`.
    .. [3] C. D. Kuglin D. C. Hines. The phase correlation image alignment
           method, Proceeding of IEEE International Conference on Cybernetics
           and Society, pp. 163-165, New York, NY, USA, 1975, pp. 163–165.
    .. [4] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    .. [5] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [6] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
    """
    if (reference_mask is not None) or (moving_mask is not None):
        shift = _masked_phase_cross_correlation(
            reference_image, moving_image, reference_mask, moving_mask, overlap_ratio
        )
        return shift, np.nan, np.nan

    # images must be the same shape
    if reference_image.shape != moving_image.shape:
        raise ValueError("images must be same shape")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = reference_image
        target_freq = moving_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = fftn(reference_image)
        target_freq = fftn(moving_image)
    else:
        raise ValueError('space argument must be "real" of "fourier"')

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = np.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError("normalization must be either phase or None")
    cross_correlation = ifftn(image_product)

    # Locate maximum
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )
    midpoint = np.array([np.fix(axis_size / 2) for axis_size in shape])

    float_dtype = image_product.real.dtype

    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(shape)[shift > midpoint]

    if upsample_factor == 1:
        src_amp = np.sum(np.real(src_freq * src_freq.conj()))
        src_amp /= src_freq.size
        target_amp = np.sum(np.real(target_freq * target_freq.conj()))
        target_amp /= target_freq.size
        CCmax = cross_correlation[maxima]
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        upsample_factor = np.array(upsample_factor, dtype=float_dtype)
        shift = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shift * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        # Locate maximum and map back to original pixel grid
        maxima = np.unravel_index(
            np.argmax(np.abs(cross_correlation)), cross_correlation.shape
        )
        CCmax = cross_correlation[maxima]

        maxima = np.stack(maxima).astype(float_dtype, copy=False)
        maxima -= dftshift

        shift += maxima / upsample_factor

        src_amp = np.sum(np.real(src_freq * src_freq.conj()))
        target_amp = np.sum(np.real(target_freq * target_freq.conj()))

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shift[dim] = 0

    if disambiguate:
        if space.lower() != 'real':
            reference_image = ifftn(reference_image)
            moving_image = ifftn(moving_image)
        shift = _disambiguate_shift(reference_image, moving_image, shift)

    # Redirect user to masked_phase_cross_correlation if NaNs are observed
    if np.isnan(CCmax) or np.isnan(src_amp) or np.isnan(target_amp):
        raise ValueError(
            "NaN values found, please remove NaNs from your "
            "input data or use the `reference_mask`/`moving_mask` "
            "keywords, eg: "
            "phase_cross_correlation(reference_image, moving_image, "
            "reference_mask=~np.isnan(reference_image), "
            "moving_mask=~np.isnan(moving_image))"
        )

    return shift, _compute_error(CCmax, src_amp, target_amp), _compute_phasediff(CCmax)