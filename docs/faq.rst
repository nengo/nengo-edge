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

Other questions?
================

If you have a question that isn't answered here, use the contact form within the
`NengoEdge application <https://edge.nengo.ai/contact-us>`_ to get in touch with us and we will
do our best to help!
