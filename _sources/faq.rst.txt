**************************
Frequently asked questions
**************************

How do I upload my own keyword spotting data?
=============================================

To upload your own data, navigate to the Datasets page.
Press the Upload new data button on the top right corner of the screen.
You must provide a name for this data,
as well as a file containing the compressed data.

The data should consist of audio files organized into ``train``/``validation``/``test``
folders, containing different data samples used for
`cross-validation <https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets>`_.
Each of those folders should contain subfolders,
where the name of the subfolder is the keyword label.
Currently supported audio files are ``.wav``, ``.flac``, and ``.mp3``.
You can also include a special ``background_noise`` folder containing ``.wav`` files
representing background noise,
which will be mixed in with the keyword audio during training.

For example:

* ``my_dataset``

  * ``train``

    * ``up``

      * ``file0.wav``

      * ``file1.wav``

    * ``down``

      * ``file2.wav``

      * ``file3.wav``

  * ``validation``

    * ``up``

      * ``file4.wav``

      * ``file5.wav``

    * ``down``

      * ``file6.wav``

      * ``file7.wav``

  * ``test``

    * ``up``

      * ``file8.wav``

      * ``file9.wav``

    * ``down``

      * ``file10.wav``

      * ``file11.wav``

  * ``background_noise``

    * ``file12.wav``

    * ``file13.wav``

This should all then be combined into a ``.tar.gz`` archive,
which is the file that you will upload to NengoEdge.

For more information, see :doc:`tutorials/uploading-datasets`.

How do I pick a good batch size?
================================

The right batch size can be difficult to choose
and can be highly dependent on other parameters.

Lower batch sizes can improve performance,
as they allow the model to perform more learning updates on the same amount of data.
However, if the batch size is too small
the model could be too sensitive to the noise in individual samples.
The learning rate may need to be lowered to keep the updates stable in this case.

On the other hand, large batch sizes have the advantage of more stable gradients, and
being able to better leverage parallel computation in GPUs to process more data in
the same amount of time.
Often increasing the learning rate alongside a batch size increase
can negate the downside of fewer updates per epoch.
Larger batches are less sensitive to noise,
which can allow them to handle larger learning rates more safely.

In addition, the batch size is practically limited by the amount of GPU memory
available. Selecting too large a batch size will cause out-of-memory errors.

What are MFCCs and the Mel spectrum?
====================================

The `Mel scale <https://en.wikipedia.org/wiki/Mel_scale>`_ is a scale of pitch that more
closely matches human perception than the frequency in Hz. It is used to convert
the audio signals into features that are more meaningful for speech processing.

`Mel-frequency cepstral coefficients (MFCCs) <https://en.wikipedia.org/wiki/Mel-frequency_cepstrum>`_
are a way of representing a sound using the Mel spectrum. They are commonly used as
features in speech models. NengoEdge will automatically compute MFCCs based on your
selected parameters (see the *Run configuration* page for details).

MFCC values may be further processed to reduce noise sensitivity using a discrete cosine
transform (DCT). NengoEdge does this by default, but you can disable this by setting the
DCT features to 0 in the *Audio preprocessing* section on the *Run configuration* page.

What are the Activation type options?
=====================================

When configuring the network architecture in NengoEdge, some layers provide a choice of
activation type. These options have been chosen to work well with the provided
architecture and they should rarely need to be changed.

``relu``, ``sigmoid``, and ``swish`` are non-linear activation functions and are
commonly used in intermediate layers, while ``linear`` and ``softmax`` are usually used
for the final output layers.

* linear: Passes the input through unmodified.
* relu: ``max(x, 0)``
* sigmoid: ``1 / (1 + exp(-x))``
* softmax: ``exp(x) / sum(exp(x))``
* swish: ``x*sigmoid(x)``

What are warmup steps?
======================

NengoEdge uses a learning rate scheduler to dynamically adjust the learning rate during
training. For ASR models this scheduler starts with a very small learning rate that
increases during the warmup phase, and then slowly decreases afterwards.

When starting a new training run, this value should usually be 25--50% of the total
training steps. When initializing training from a previously trained
run this value can be much smaller; a good starting point would be around 5% of the
training steps.

Consider increasing the warmup steps if the model performance is oscillating or
overfitting early on.

Other questions?
================

If you have a question that isn't answered here, use the contact form within the
`NengoEdge application <https://edge.nengo.ai/contact-us>`_ to get in touch with us and
we will do our best to help!
