************
Installation
************

NengoEdge models use the `TensorFlow <https://www.tensorflow.org/>`_
machine learning library. If you already have TensorFlow installed,
then all you need is to::

  pip install nengo-edge

If you do not have TensorFlow installed, see the instructions below.

TensorFlow installation
=======================

Installing TensorFlow can be done a few ways.
If you are not sure which to use and you have an Nvidia GPU,
we recommend using Mamba.

.. tabs::

   .. tab:: Mamba

      Mamba is the best choice if you are starting from scratch
      and want your NengoEdge models to run on your Nvidia GPU.

      1. Download the
         `Mambaforge installer <https://mamba.readthedocs.io/en/latest/installation.html#fresh-install>`_
         for your OS and follow the installation instructions on that page.

      2. Open a terminal or command-line prompt
         and create an environment for NengoEdge::

           mamba create -n nengo-edge python=3.10
           mamba activate nengo-edge

      3. Install TensorFlow::

           mamba install tensorflow

   .. tab:: Conda

      If you already have familiarity with Conda, it can be used
      to run NengoEdge models by installing from the ``conda-forge`` channel.

      1. Download the
         `Miniconda installer <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`_
         for your OS and follow the installation instructions on that page.

      2. Open a terminal or command-line prompt and ensure conda is up to date.
         Configure it to use conda-forge by default::

           conda update conda
           conda config --add channels conda-forge
           conda config --set channel_priority strict

      3. Create an environment for NengoEdge::

           conda create -n nengo-edge python=3.10
           conda activate nengo-edge

      4. Install TensorFlow::

           conda install tensorflow

   .. tab:: pip

      If you do not have an Nvidia GPU or do not want to run NengoEdge models
      on your GPU, ``pip`` is the simplest way to get started.

      We recommend first setting up a virtual environment using
      `venv <https://docs.python.org/3/library/venv.html>`_ and installing
      all packages in that virtual environment.

      Whether in a venv or not, the command to install TensorFlow is the same::

        pip install tensorflow
