# Originally based on
# https://github.com/google-coral/project-keyword-spotter/blob/master/model.py
# https://github.com/google-coral/project-keyword-spotter/blob/master/mel_features.py
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract MFCC speech features from audio data.

This is designed to work with minimal python dependencies (numpy only) so that it
can run on on-device CPUs.
"""


from typing import Union

import numpy as np


def frame(data: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
    """
    Convert array into a sequence of successive possibly overlapping frames.

    An n-dimensional array of shape (batch_size, num_samples, ...) is converted into an
    (n+1)-D array of shape (batch_size, num_frames, window_length, ...), where each
    frame starts hop_length points after the preceding one.

    This is accomplished using stride_tricks, so the original data is not
    copied.  However, there is no zero-padding, so any incomplete frames at the
    end are not included.

    Parameters
    ----------
    data : np.array
        Data to be split into frames, with shape ``(batch_size, num_samples, ...)``.
    window_length : int
        Number of samples in each frame.
    hop_length : int
        Advance (in samples) between each frame.

    Returns
    -------
    frames : np.ndarray
        (N+1)-D np.array with as many rows as there are complete frames that can be
        extracted.
    """
    num_samples = data.shape[1]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (data.shape[0], num_frames, window_length) + data.shape[2:]
    strides = data.strides[:1] + (data.strides[1] * hop_length,) + data.strides[1:]
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length: int) -> np.ndarray:
    """
    Calculate a "periodic" Hann window.

    The classic Hann window is defined as a raised cosine that starts and
    ends on zero, and where every value appears twice, except the middle
    point for an odd-length window.  Matlab calls this a "symmetric" window
    and np.hanning() returns it.  However, for Fourier analysis, this
    actually represents just over one cycle of a period N-1 cosine, and
    thus is not compactly expressed on a length-N Fourier basis.  Instead,
    it's better to use a raised cosine that ends just before the final
    zero value - i.e. a complete cycle of a period-N cosine.  Matlab
    calls this a "periodic" window. This routine calculates it.

    Parameters
    ----------
    window_length : int
        The number of points in the returned window.

    Returns
    -------
    window : np.ndarray
        A 1D array containing the periodic hann window.
    """
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))


def stft_magnitude(
    signal: np.ndarray, fft_length: int, hop_length: int, window_length: int
) -> np.ndarray:
    """
    Calculate the short-time Fourier transform magnitude.

    Parameters
    ----------
    signal : np.ndarray
        2D np.array of the batched input time-domain signal.
    fft_length : int
        Size of the FFT to apply.
    hop_length : int
        Advance (in samples) between each frame passed to FFT.
    window_length : int
        Length of each block of samples to pass to FFT.

    Returns
    -------
    spectrogram : np.ndarray
        3D np.array where the first axis is the batch, the second axis is the number
        of frames, and the third axis contains the magnitudes of the fft_length/2+1
        unique values of the FFT for the corresponding frame of input samples.
    """
    frames = frame(signal, window_length, hop_length)
    # Apply frame window to each frame. We use a periodic Hann (cosine of period
    # window_length) instead of the symmetric Hann of np.hanning (period
    # window_length-1).
    window = periodic_hann(window_length)
    windowed_frames = frames * window[None, :]
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


def hertz_to_mel(
    frequencies_hertz: "Union[float, np.ndarray]",
) -> "Union[float, np.ndarray]":
    """
    Convert frequencies to mel scale using HTK formula.

    Parameters
    ----------
    frequencies_hertz : Union[float, np.ndarray]
        Scalar or np.array of frequencies in hertz.

    Returns
    -------
    mel_values : Union[float, np.ndarray]
        Object of same size as ``frequencies_hertz`` containing corresponding values
        on the mel scale.
    """

    # Mel spectrum constants and functions.
    _MEL_BREAK_FREQUENCY_HERTZ = 700.0
    _MEL_HIGH_FREQUENCY_Q = 1127.0

    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
    )


def spectrogram_to_mel_matrix(
    num_mel_bins: int = 20,
    num_spectrogram_bins: int = 129,
    audio_sample_rate: int = 8000,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3800.0,
) -> np.ndarray:
    """
    Return a matrix that can post-multiply spectrogram rows to make mel.

    Returns a np.array matrix A that can be used to post-multiply a matrix S of
    spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
    "mel spectrogram" M of frames x num_mel_bins.  M = S A.

    The classic HTK algorithm exploits the complementarity of adjacent mel bands
    to multiply each FFT bin by only one mel weight, then add it, with positive
    and negative signs, to the two adjacent mel bands to which that bin
    contributes.  Here, by expressing this operation as a matrix multiply, we go
    from num_fft multiplies per frame (plus around 2*num_fft adds) to around
    num_fft^2 multiplies and adds.  However, because these are all presumably
    accomplished in a single call to np.dot(), it's not clear which approach is
    faster in Python.  The matrix multiplication has the attraction of being more
    general and flexible, and much easier to read.

    Parameters
    ----------
    num_mel_bins : int
        How many bands in the resulting mel spectrum.  This is
        the number of columns in the output matrix.
    num_spectrogram_bins : int
        How many bins there are in the source spectrogram
        data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
        only contains the nonredundant FFT bins.
    audio_sample_rate : int
        Samples per second of the audio at the input to the
        spectrogram. We need this to figure out the actual frequencies for
        each spectrogram bin, which dictates how they are mapped into mel.
    lower_edge_hertz : float
        Lower bound on the frequencies to be included in the mel
        spectrum.  This corresponds to the lower edge of the lowest triangular
        band.
    upper_edge_hertz : float
        The desired top edge of the highest frequency band.

    Returns
    -------
    mel_matrix : np.ndarray
        An array with shape (num_spectrogram_bins, num_mel_bins) that can post-multiply
        the spectrogram to compute mel values.

    Raises
    ------
    ValueError
        If frequency edges are incorrectly ordered or out of range.
    """
    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError(f"lower_edge_hertz {lower_edge_hertz:.1f} must be >= 0")
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            f"lower_edge_hertz {lower_edge_hertz:.1f} >= upper_edge_hertz "
            f"{upper_edge_hertz:.1f}"
        )
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError(
            f"upper_edge_hertz {upper_edge_hertz:.1f} is greater than Nyquist "
            f"{nyquist_hertz:.1f}"
        )
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2
    )
    # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
    # of spectrogram values.
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (
            center_mel - lower_edge_mel
        )
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (
            upper_edge_mel - center_mel
        )
        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    # HTK excludes the spectrogram DC bin; make sure it always gets a zero
    # coefficient.
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def dct_matrix(input_features: int, output_features: int) -> np.ndarray:
    """Returns a matrix that can post-multiply mel features to compute DCT features."""

    matrix = 2.0 * np.cos(
        np.pi
        * np.outer(
            np.arange(input_features) * 2.0 + 1.0,
            np.arange(output_features),
        )
        / (2.0 * input_features)
    )
    # DCT normalization
    matrix *= 1.0 / np.sqrt(2.0 * input_features)

    return matrix


class LogMelFeatureExtractor:
    """
    Compute log mel spectrogram slices from audio data.

    Parameters
    ----------
    window_size_ms : float
        Length of each spectrogram time slice (in milliseconds).
    window_stride_ms : float
        Length of time (in milliseconds) between spectrogram slices.
    mel_num_bins : int
        The number of Mel bins.
    dct_num_features : int
        The number of DCT features.
    sample_rate : int
        Sample rate (in Hz) of the audio data.
    mel_lower_edge_hertz : float
        Lower bound on the frequencies to be included in the mel spectrum.
    mel_upper_edge_hertz : float
        Upper bound on the frequencies to be included in the mel spectrum.
    log_epsilon : float
        Small value added to avoid NaNs in log calculation.
    """

    def __init__(
        self,
        window_size_ms: float,
        window_stride_ms: float,
        mel_num_bins: int,
        dct_num_features: int,
        sample_rate: int,
        mel_lower_edge_hertz: float,
        mel_upper_edge_hertz: float,
        log_epsilon: float,
    ):
        self._norm_factor = 3
        self.window_size_samples = int(round(sample_rate * window_size_ms / 1000.0))
        self.window_stride_samples = int(round(sample_rate * window_stride_ms / 1000.0))
        self.mel_num_bins = mel_num_bins
        self.dct_num_features = dct_num_features
        self.fft_length = 2 ** int(
            np.ceil(np.log(self.window_size_samples) / np.log(2.0))
        )
        self.log_epsilon = log_epsilon

        self.mel_matrix = (
            spectrogram_to_mel_matrix(
                num_mel_bins=mel_num_bins,
                num_spectrogram_bins=self.fft_length // 2 + 1,
                audio_sample_rate=sample_rate,
                lower_edge_hertz=mel_lower_edge_hertz,
                upper_edge_hertz=mel_upper_edge_hertz,
            ),
        )

        if dct_num_features:
            self.dct_matrix = dct_matrix(mel_num_bins, dct_num_features)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Computes spectrograms on a batch of input data (not streaming).

        Parameters
        ----------
        data : np.ndarray
            Input data with shape ``(batch_size, audio_samples)``.

        Returns
        -------
        mfccs : np.ndarray
            Output mel spectrograms with shape
            ``(batch_size, frames, dct_num_features)`` (or ``(..., mel_num_features)``
            if ``dct_num_features==0``).
        """

        # compute windowing + fft
        features = stft_magnitude(
            data,
            fft_length=self.fft_length,
            hop_length=self.window_stride_samples,
            window_length=self.window_size_samples,
        )

        # compute mel features
        features = np.matmul(features, self.mel_matrix)

        # compute log mel features
        features = np.log(np.maximum(features, self.log_epsilon))

        # compute dct
        if self.dct_num_features:
            features = np.matmul(features, self.dct_matrix)

        return features
