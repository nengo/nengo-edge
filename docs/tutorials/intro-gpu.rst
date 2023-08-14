Fast keyword detection with LMUs on GPU
=======================================

.. raw:: html

   <link rel="canonical" href="https://appliedbrainresearch.com/blog/fast-keyword-detection-with-lmus-on-gpu" />

The Legendre Memory Unit (LMU) algorithm is a new alternative to LSTMs and Transformers
with some `key advantages <https://arxiv.org/abs/2009.04465>`__,
especially at the edge.
NengoEdge makes it easy to build LMU models with any dataset
and deploy those models anywhere.
In this article, we'll show how easy it is to use NengoEdge
to deploy a keyword detection model
using LMUs on a general purpose Nvidia GPU.

The goal of our model will be to add voice controls
to a device that moves along two dimensions, like a plotter.
We want to be able to say what direction to move,
and to be able to stop it
when it reaches the desired location by saying "stop."

Set up your dataset
-------------------

The first step is to set up your dataset.
In NengoEdge, you can upload your own raw audio files,
or use the SpeechCommands audio files
that are available when you create your account.
Raw audio files are used to make Datasets,
which let you specify how to use those raw audio files
in your machine learning pipeline.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/639a796ab1537b601940bf7b_YaoCQLh6A6OnIxeFw3nqs-vM.png

1. Click on **Datasets** in the top bar.

2. Under the **SpeechCommands v2** heading, click **New dataset**.

3. Give the dataset a name.
   We will use the name "Directions"

4. Set the dataset parameters.
   We will select the keywords "up, down, left, right, stop"

5. Click **Create**.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/639a796a570a4d9f563cc039_rykUVsArevhKk82ENHqOz9M_.png

*If you're unsure of what a parameter does,
hovering over it will display a tooltip with more information.*

Find a successful test run
--------------------------

We recommend doing at least one shorter test run
to make sure that your longer run will yield good results.
If you're using your own raw audio files,
you may need several test runs to find a good set of parameters.

*Training time can get expensive,
so shorter test runs save money in the long run.
You can reuse the weights from the short run as the starting point for longer runs,
so there's no downside to a test run.*

1. Click on **Runs** in the top bar.

2. Click on **Create new project** to organize the runs for this project.
   We will use the name "Directions"

3. Under the Directions heading, click **+KWS** to create a new keyword spotting run.

4. Set the run parameters.

   - We will use the name "Directions test 1"
   - We will use the LMU-large network to leverage the power of LMUs
   - We will set the number of training steps to 2000
   - We will set the validation interval to 100

5. Click the Optimize button.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/639a796a3155c4b020fc1963_sJmE68vkvqHYx2cHkm6O-A16.png

These parameter will result in a run
that finishes in around an hour,
and will give more feedback on performance in these early training stages.

The **Progress** tab gives you a rough idea of whether your test run is successful.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/63f7d1ad6c0317becb773a70_G3aHyh-J5L1YSrVdH_mr63CC.png

In this case, our loss is going down steadily,
and the validation accuracy reaches a good level,
so we will consider this a successful test run.

*With the built in SpeechCommands v2 dataset,
we expect performance to be very good using default run parameters.
However, using your own dataset may require parameter tweaking.*

If this wasn't the case, we can look at the **Results** tab
to see where the model is underperforming to guide future test runs.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/63f7d1dd7ddcec5dfa353563_ISzW173mW1Hj5L77x_0o5z1r.png

The confusion matrix can identify specific labels (words)
that are not being classified well,
which can help direct data collection when using your own dataset.
General accuracy metrics can help you identify other issues.
Experiment with the run parameters
to see how each parameter affects performance metrics
if your accuracy numbers are too low.

Set up a full training run
--------------------------

Once you are satisfied with your test run,
create a new run to train to convergence.

1. Click on **Runs** in the top bar.

2. Under the **Directions** heading, click **+KWS**.

3. Set the name to "Directions full 1"

4. Click the drop-down next to **Initial weights**.
   Select the successful test run, "Directions test 1".

5. Set other run parameters.

   - We will set the number of training steps to 20000
   - We will set the validation interval to 500

6. Click the **Optimize** button.

These parameter will result in a run that finishes in several hours,
and will give coarse feedback compared to the test runs.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/63f7d21850631f96e18206c3_NllK0fPMJ58CEMQ1MdMm3fiA.png

You will see on the **Progress** page that initial performance
will start around where the test run finished off.
By the end of the optimization process, performance should be even better.
In our case, we started at 93.94% accuracy and finished at 97.85% accuracy.

*If you want to deploy this model to an edge device,
you can stop here and go to the tutorial for your target device.*

Test and deploy the trained model
---------------------------------

Your model is now ready to be deployed to your environment
and used either by itself or as part of a larger system.
Steps for integration will depend on your setup,
but here we will show how to do a simple deployment
using :doc:`NengoEdge tools <../developers>` and
the `SavedModel <https://www.tensorflow.org/guide/saved_model>`__ format.

NengoEdge tools is a Python package that assists in taking a trained model
exported from NengoEdge and deploying it in your own application.
We will use it to deploy the model we trained,
which we will export in SavedModel format.

.. image:: https://uploads-ssl.webflow.com/635051b5ae517f8542a06c6f/63f7d25746323435ba3374bd_sx4j0NyGm8SxaBWJbllHEBQw.png

1. Install NengoEdge tools following :doc:`these steps <../developers>`.

2. Download the :doc:`live microphone demo <../examples/microphone-demo/microphone-demo>`
   to a local directory.

3. Open NengoEdge to the main **Runs** page.

4. Click the **Deploy** icon next to the fully trained run.

5. Under the **Download model** heading,
   click the **Format** dropdown and select **Binary**.

6. Click the **Download** button.

7. Extract the downloaded ``artifact.zip``
   to the same directory as the live microphone demo.

8. Open the demo with ``jupyter notebook /path/to/microphone-demo.ipynb``

9. Edit the notebook to use the labels defined in the dataset.

   ``labels = ["<silence>", "<unknown>", "up", "down", "left", "right", "stop"]``

After running each cell in the notebook,
you should see the message "Press Enter to quit,"
at which point you can speak into your microphone
to see your model classify keywords in real time.

----

*This tutorial was originally posted on*
|appliedbrainresearch.com|_.

.. |appliedbrainresearch.com| replace:: *appliedbrainresearch.com*
.. _appliedbrainresearch.com: https://appliedbrainresearch.com/blog/fast-keyword-detection-with-lmus-on-gpu
