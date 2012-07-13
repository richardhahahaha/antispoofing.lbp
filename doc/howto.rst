.. vim: set fileencoding=utf-8 :
.. Ivana Chingovska <ivana.chingovska@idiap.ch>
.. Mon Jul  9 19:36:36 CEST 2012

================================================
 Workflow for LBP based spoofing counter measure
================================================

This document explains the workflow for the counter measures based on Local Binary Patterns (LBP) and how to replicate the results obtained at::

  @INPROCEEDINGS{Chingovska_BIOSIG_2012,
  author = {Chingovska, Ivana and Anjos, Andr{\'{e}} and Marcel, S{\'{e}}bastien},
  keywords = {Attack, Counter-Measures, Counter-Spoofing, Face Recognition, Liveness Detection, Replay, Spoofing},
  month = sep,
  title = {On the Effectiveness of Local Binary Patterns in Face Anti-spoofing},
  journal = {IEEE BIOSIG 2012},
  year = {2012},
  }
 
Namely, the document explains how to use the package in order to: a) calculate the LBP features on the REPLAY-ATTACK database; b) perform classification using Chi-2, Linear Discriminant Analysis (LDA) and Support Vector Machines (SVM).

It is assumed you have followed the installation instructions for the package,
as described in the ``README.rst`` file located in the root of the package and
got this package installed and the REPLAY-ATTACK database downloaded and
uncompressed in a directory.  After running the ``buildout`` command, you
should have all required utilities sitting inside the ``bin`` directory. We
expect that the video files downloaded for the PRINT-ATTACK database are
installed in a sub-directory called ``database`` at the root of the package. 
You can use a link to the location of the database files, if you don't want to
have the database installed on the root of this package::

  $ ln -s /path/where/you/installed/the/replay-attack-database database

If you don't want to create a link, use the ``--input-dir`` flag (available in all the scripts) to specify the root directory containing the database files. That would be the directory that *contains* the sub-directories ``train``, ``test``, ``devel`` and ``face-locations``.


I. CALCULATE THE LBP FEATURES
-----------------------------

The first stage of the process is calculating the feature vectors, which are essentially normalized LBP histograms. There are two types of feature vectors:
1. per-video averaged feature-vectors (the normalized LBP histograms for each frame, averaged over all the frames of the video. The result is a single feature vector for the whole video), or
2. a single feature vector for each frame of the video (saved as a multiple row array in a single file). 

The program to be used for the first case is `script/calclbp.py`, and for the second case `script/calcframelbp.py`. They both uses the utility script    
`spoof/calclbp.py`. Depending on the command line arguments, they can compute different types of LBP histograms over the normalized face bounding box. Furthermore, the normalized face-bounding box can be divided into blocks or not.

The following command will calculate the per-video averaged feature vectors of all the videos in the REPLAY-ATTACK database and will put the resulting .hdf5 files with the extracted feature vectors in the default output directory `./lbp_features`.

.. code-block:: shell

  $ ./bin/calclbp.py --ff 50

In the above command, the face size filter is set to 50 pixels (as in the paper), and the program will discard all the frames with detected faces smaller then 50 pixels as invalid.

To see all the options for the scripts `calclbp.py` and `calcframelbp.py`, just type `--help` at the command line. Change the default option in order to obtain various features, as described in the paper. 


II. CLASSIFICATION USING CHI-2 DISTANCE
---------------------------------------

The clasification using Chi-2 distance consists of two steps. The first one is creating the histogram model (average LBP histogram of all the real access videos in the training set). The second step is comparison of the features of development and test videos to the model histogram and writing the results.

The script to use for creating the histogram model is `script/mkhistmodel.py`. It expects that the LBP features of the videos are stored in a folder `./lbp_features`. The model histogram will be written in the default output folder `./res`. You can change this default features by setting the input arguments. To execute this script, just run:

.. code-block:: shell

  $ ./bin/mkhistmodel.py

The script for performing Chi-2 histogram comparison is `script/cmphistmodels.py`, and it assumes that the model histogram has been already created. It makes use of the utility script `spoof/chi2.py` and `ml/perf.py` for writing the results in a file. The default input directory is `./lbp_features`, while the default input directoru for the histogram model as well as default output directory is `./res`. To execute this script, just run: 

.. code-block:: shell

  $ ./bin/cmphistmodel.py

To see all the options for the scripts `mkhistmodel.py` and `cmphistmodels.py`, just type `--help` at the command line.


III. CLASSIFICATION WITH LINEAR DISCRIMINANT ANALYSIS (LDA)
-----------------------------------------------------------

The classification with LDA is performed using the script `script/ldatrain_lbp.py`. It makes use of the scripts `ml/lda.py`, `ml\pca.py` (if PCA reduction is performed on the data) and `ml\norm.py` (if the data need to be normalized). The default input and output directories are `./lbp_features` and `./res`. To execute the script with prior PCA dimensionality reduction as is done in the paper, call:

.. code-block:: shell

  $ ./bin/ldatrain_lbp.py -r 

To see all the options for this script, just type `--help` at the command line.


IV. CLASSIFICATION WITH SUPPORT VECTOR MACHINE (SVM)
----------------------------------------------------

The classification with SVM is performed using the script `script/svmtrain_lbp.py`. It makes use of the scripts `ml\pca.py` (if PCA reduction is performed on the data) and `ml\norm.py` (if the data need to be normalized). The default input and output directories are `./lbp_features` and `./res`. To execute the script with prior normalization of the data in the range [-1, 1] as in the paper, the default parameters, call:

.. code-block:: shell

  $ ./bin/svmtrain_lbp.py -n

To see all the options for this script, just type `--help` at the command line.


Problems
--------

In case of problems, please contact ``ivana.chingovska@idiap.ch`` and/or ``andre.anjos@idiap.ch``.
