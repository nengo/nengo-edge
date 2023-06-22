import json

import pytest


@pytest.fixture
def param_dir(tmp_path):
    params = {
        "preprocessing": {
            "window_size_ms": 40,
            "window_stride_ms": 20,
            "mel_num_bins": 40,
            "dct_num_features": 20,
            "sample_rate": 16000,
            "mel_lower_edge_hertz": 20,
            "mel_upper_edge_hertz": 7000,
            "log_epsilon": 1e-12,
        },
        "model": {
            "input_shape": [1, 20],
            "output_shape": [1, 10],
            "state_shapes": [10, 10],
            "return_sequences": True,
        },
    }

    with open(tmp_path / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f)

    return tmp_path
