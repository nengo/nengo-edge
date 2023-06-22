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
   - Deprecated
   - Removed
   - Fixed

23.2.24 (unreleased)
====================

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

.. _#4: https://github.com/nengo/nengo-edge/pull/4

23.2.23 (February 23, 2023)
===========================

**Fixed**

- Fixed an issue causing pip to refuse to install ``nengo-edge``. (`#3`_)

.. _#3: https://github.com/nengo/nengo-edge/pull/3

23.1.31 (January 31, 2023)
==========================

Initial release
