***************
Release history
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Fixed
   - Deprecated
   - Removed

23.9.28 (unreleased)
====================

**Added**

- Updated documentation with new content (additional tutorials, CLI documentation,
  FAQ section). (`#9`_)
- Added support for input string processing and NLP model inference in 
  ``SavedModelRunner``. (`#17`_)
- Added ``NetworkTokenizer`` class to perform remote calls to a device
  CLI that supports sentencepiece tokenization. (`#17`_)

**Changed**

- ``SavedModelRunner`` tokenizer now uses ``SentencepieceTokenizer`` instead of
  ``FastSentencepieceTokenizer`` to ensure compatibility with the core sentencepiece 
  library. (`#17`_)

**Fixed**

- Fixed model output decoding for ASR. ``SavedModelRunner`` now removes blank
  tokens and merges repeating tokens before detokenization. (`#17`_)

.. _#9: https://github.com/nengo/nengo-edge/pull/9
.. _#17: https://github.com/nengo/nengo-edge/pull/17

23.9.27 (September 27, 2023)
============================

**Added**

- Added warning when a downloaded nengo-edge model artifacts' version does not 
  match local environment nengo-edge version. (`#6`_)
- Added ``nengo-edge package-dataset`` CLI command, which can be used to validate
  and package KWS and ASR datasets. (`#10`_)
- ``SavedModelRunner.run`` now automatically decodes ASR model outputs via the exported 
  sentencepiece tokenizer. (`#15`_)

**Changed**

- ``SavedModelRunner`` now uses Keras' SavedModel format, instead of the raw
  TensorFlow SavedModel format. (`#8`_)
- ``SavedModelRunner`` can now take ragged ``object``-arrays as input. (`#8`_)
- TensorFlow is now an optional dependency installed with 
  ``pip install nengo-edge[optional]``. (`#13`_)

.. _#6: https://github.com/nengo/nengo-edge/pull/6
.. _#8: https://github.com/nengo/nengo-edge/pull/8
.. _#10: https://github.com/nengo/nengo-edge/pull/10
.. _#13: https://github.com/nengo/nengo-edge/pull/13
.. _#15: https://github.com/nengo/nengo-edge/pull/15
.. _#17: https://github.com/nengo/nengo-edge/pull/17


23.7.30 (July 30, 2023)
=======================

**Added**

- Added ``CoralRunner`` for running models exported for the Coral board. (`#4`_)
- Added ``DiscoRunner`` for running models exported for the Disco board. (`#4`_)
- Added ``NordicRunner`` for running models exported for the Nordic board. (`#4`_)
- Added on-device MFCC extraction code
  (``device_modules.np_mfcc.LogMelFeatureExtractor``). (`#4`_)
- Added two new examples demonstrating how to run models exported for the
  Coral/Disco/Nordic devices. (`#4`_)

**Changed**

- Renamed ``tflite_runner.Runner`` to ``TFLiteRunner``. (`#4`_)
- Renamed ``saved_model_runner.Runner`` to ``SavedModelRunner``. (`#4`_)
- ``TFLiteRunner.reset_state`` now takes a ``batch_size`` argument, which can be used
  to prepare the model to run with different batch sizes. (`#5`_)

.. _#4: https://github.com/nengo/nengo-edge/pull/4
.. _#5: https://github.com/nengo/nengo-edge/pull/5

23.2.23 (February 23, 2023)
===========================

**Fixed**

- Fixed an issue causing pip to refuse to install ``nengo-edge``. (`#3`_)

.. _#3: https://github.com/nengo/nengo-edge/pull/3

23.1.31 (January 31, 2023)
==========================

Initial release
