# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf
from nengo_edge_models.audio import SpeechFeatures
from nengo_edge_models.audio.base_data import MFCCOptions

from nengo_edge.device_modules import np_mfcc


def test_frame(rng: np.random.RandomState) -> None:
    x = rng.uniform(0, 1, size=(32, 16000))

    y0 = tf.signal.frame(x, 400, 160)

    y1 = np_mfcc.frame(x, 400, 160)

    assert np.allclose(y0, y1)


def test_hann(rng: np.random.RandomState) -> None:
    y0 = tf.signal.hann_window(400, periodic=True, dtype=tf.float64)
    y1 = np_mfcc.periodic_hann(400)

    assert np.allclose(y0, y1)


def test_mel_matrix() -> None:
    y0 = tf.signal.linear_to_mel_weight_matrix(dtype=tf.float64)
    y1 = np_mfcc.spectrogram_to_mel_matrix()

    assert np.allclose(y0, y1)


def test_spectrogram(rng: np.random.RandomState) -> None:
    x = rng.uniform(0, 1, size=(32, 16000))

    options = MFCCOptions(
        window_size_ms=400,
        window_stride_ms=160,
        sample_rate=1000,
        dct_num_features=0,
    )
    layer = SpeechFeatures(options)
    y0 = layer.spectrogram(x)

    y1 = np_mfcc.stft_magnitude(
        x,
        fft_length=2 ** int(np.ceil(np.log(400) / np.log(2.0))),
        hop_length=160,
        window_length=400,
    )

    assert np.allclose(y0, y1), np.max(abs(y0 - y1))


def test_mel_spectrogram(rng: np.random.RandomState) -> None:
    x = rng.uniform(0, 1, size=(32, 16000))

    options = MFCCOptions()

    layer = SpeechFeatures(options, dtype="float64")
    y0 = layer(x)

    feature_extractor = np_mfcc.LogMelFeatureExtractor(
        window_size_ms=options.window_size_ms,
        window_stride_ms=options.window_stride_ms,
        mel_num_bins=options.mel_num_bins,
        dct_num_features=options.dct_num_features,
        sample_rate=options.sample_rate,
        mel_lower_edge_hertz=options.mel_lower_edge_hertz,
        mel_upper_edge_hertz=options.mel_upper_edge_hertz,
        log_epsilon=options.log_epsilon,
    )
    y1 = feature_extractor(x)

    assert np.allclose(y0, y1), np.max(abs(y0 - y1))


def test_dct(rng: np.random.RandomState) -> None:
    x = rng.uniform(0, 1, size=(32, 49, 40))
    x = tf.cast(x, tf.float32)
    layer = SpeechFeatures(MFCCOptions(dct_num_features=20))
    layer.build(x.shape)
    y0 = layer.dct(x)

    dct_matrix = np_mfcc.dct_matrix(40, 20)
    y1 = np.matmul(x, dct_matrix)

    assert np.allclose(y0, y1, atol=1e-5), np.max(abs(y0 - y1))


def test_mfcc_features(rng: np.random.RandomState) -> None:
    options = MFCCOptions()
    x = rng.uniform(0, 2, size=(32, 16000))

    feature_layer = SpeechFeatures(options, dtype=tf.float64)
    y0 = feature_layer(x)

    feature_extractor = np_mfcc.LogMelFeatureExtractor(
        window_size_ms=options.window_size_ms,
        window_stride_ms=options.window_stride_ms,
        mel_num_bins=options.mel_num_bins,
        dct_num_features=options.dct_num_features,
        sample_rate=options.sample_rate,
        mel_lower_edge_hertz=options.mel_lower_edge_hertz,
        mel_upper_edge_hertz=options.mel_upper_edge_hertz,
        log_epsilon=options.log_epsilon,
    )
    y1 = feature_extractor(x)

    assert np.allclose(y0, y1), np.max(abs(y0 - y1))


def test_errors() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        np_mfcc.spectrogram_to_mel_matrix(lower_edge_hertz=-1)

    with pytest.raises(ValueError, match=">= upper_edge_hertz"):
        np_mfcc.spectrogram_to_mel_matrix(lower_edge_hertz=1, upper_edge_hertz=1)

    with pytest.raises(ValueError, match="greater than Nyquist"):
        np_mfcc.spectrogram_to_mel_matrix(upper_edge_hertz=4001, audio_sample_rate=8000)
