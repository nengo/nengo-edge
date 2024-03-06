Use your own keyword spotting data in NengoEdge
===============================================

.. raw:: html

   <link rel="canonical" href="https://appliedbrainresearch.com/blog/use-your-own-audio-data-in-nengoedge" />

If you want to train a model to detect custom keywords,
you can upload your own labelled audio datasets to NengoEdge. Note that here we'll be
specifically discussing the keyword spotting (KWS) application type; other applications
have their own dataset formats, see the Dataset upload page for each application type
for more details.

Preparing your data
-------------------

Datasets in NengoEdge are split into
three main folders named train, validation and test.
Inside each of these main folders,
labelled data is stored as ``.wav``, ``.flac`` or ``.mp3`` files
organized into sub-folders,
where the name of each sub-folder is the keyword label.
Place all audio samples associated with that label
into the corresponding sub-folder,
then create a ``.tar.gz`` archive containing
your train, validation and test folders.

Example
-------

In this example, we will use some of the dataset collected for training
the `Loihi keyword spotter <https://www.youtube.com/watch?v=nIsK7dSXBo0>`__.

This dataset was originally organized into train, validation and test folders,
with subfolders for each speaker.
Each subfolder contains ``.wav`` files with structured filenames
containing the speaker ID, keyword, and time at which the sample was recorded.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/6517119fd0b145049cb84fee_Screenshot%20from%202023-09-29%2014-03-12.png

In order to make this dataset compatible with NengoEdge,
we wrote a short Python script
(`organize-data.py <https://forum.nengo.ai/t/uploading-your-own-data-to-nengoedge/2527>`__)
to copy files around into the NengoEdge format,
and ensure that all wav files can be loaded correctly.
Feel free to use this script as a starting point for organizing your data!

After running this script on the above data, we now have the directory structure that NengoEdge expects.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/651711f2f6b267c58614fe72_Screenshot%20from%202023-09-29%2014-05-25.png

We then create a compressed archive of this directory structure. We can do this using
the :doc:`nengo-edge package-dataset </developers>` command.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/65411179dfcf95515d52d429_package-dataset.png

We are now ready to upload this archive to NengoEdge.

Uploading your data
-------------------

To upload your data, navigate to the **Datasets** page, accessible from the top navbar.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/6516f9a8f27e68c575c42d27_im4-100p.png

Click the **Upload new data** button and you will see a window pop up.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/6516faabecea7439208c8fd4_Screenshot%20from%202023-09-29%2012-26-12.png

Give your data a name and select the file on your hard drive.
Click the **Upload** button.

You will see a box with a progress bar in the bottom right of NengoEdge.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/640b95ff17d12b0c61737154_RnIsYom5lFvXvHi1kbGBZiLT.png

Once the upload is complete, your data will appear in the datasets list,
though there is no dataset associated with it yet.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/6516fb4d9034ed7a6a7057fc_Screenshot%20from%202023-09-29%2012-28-53.png

Creating new datasets from your data
------------------------------------

Uploading raw data gives you the ability to
create multiple datasets for use in training runs.
To create a new dataset, click the **New dataset** button
in the sections associated with your raw data on the **Datasets** page.

With the example dataset from above,
we created a dataset to classify the "aloha" keyword.
All keywords not selected are grouped under the "unknown" label.
Notice that the sample rate and clip duration are filled in according to the data,
but can be changed if desired.
Similarly, you can change the percentages associated with
the amount of silence or unknown samples.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/6516fbb9af840aed4164ec5a_Screenshot%20from%202023-09-29%2012-30-43.png

The dataset you created can now be used in a run.
See other tutorials like :doc:`intro-gpu`
to see how your dataset can be used to train a model.

----

*This tutorial was originally posted on*
|appliedbrainresearch.com|_.

.. |appliedbrainresearch.com| replace:: *appliedbrainresearch.com*
.. _appliedbrainresearch.com: https://appliedbrainresearch.com/blog/use-your-own-audio-data-in-nengoedge
