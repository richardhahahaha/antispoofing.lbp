Counter-Measures for Face Spoofing Attacks based on Local Binary Patterns (LBP)
===============================================================================

This package implements the LBP counter-measure to spoofing attacks to
face recognition systems as described at the paper::

  @INPROCEEDINGS{Chingovska_BIOSIG_2012,
  author = {Chingovska, Ivana and Anjos, Andr{\'{e}} and Marcel, S{\'{e}}bastien},
  keywords = {Attack, Counter-Measures, Counter-Spoofing, Face Recognition, Liveness Detection, Replay, Spoofing},
  month = sep,
  title = {On the Effectiveness of Local Binary Patterns in Face Anti-spoofing},
  journal = {IEEE BIOSIG 2012},
  year = {2012},
  }
 
The dataset used in the paper is REPLAY-ATTACK database and it is publicly available. It should be downloaded and
installed **prior** to using the programs described in this package. Visit
`the REPLAY-ATTACK database page <https://www.idiap.ch/dataset/printattack>`_ for more information.

To run the code in this package, you will also need `Bob, an open-source
toolkit for Signal Processing and Machine Learning
<http://idiap.github.com/bob>`_. The code has been tested to work with Bob
1.0.x.

Installation
------------

To follow these instructions locally you will need a local copy of this
package. Start by cloning this project with something like (shell commands are marked with a
``$`` signal)::

  $ git clone --depth=1 https://github.com/bioidiap/antispoofing.lbp.git
  $ cd antispoofing.lbp
  $ rm -rf .git # you don't need the git directories...

Alternatively, you can use the github tarball API to download the package::

  $ wget --no-check-certificate https://github.com/bioidiap/antispoofing.lbp/tarball/master -O- | tar xz 
  $ mv bioidiap-antispoofing-* antispoofing.lbp

Installation of the toolkit uses the `buildout <http://www.buildout.org/>`_
build environment. You don't need to understand its inner workings to use this
package. Here is a recipe to get you started::
  
  $ python bootstrap.py
  $ ./bin/buildout

These 2 commands should download and install all non-installed dependencies and
get you a fully operational test and development environment.

.. note::

  The python shell used in the first line of the previous command set
  determines the python interpreter that will be used for all scripts developed
  inside this package. Because this package makes use of `Bob
  <http://idiap.github.com/bob>`_, you must make sure that the ``bootstrap.py``
  script is called with the **same** interpreter used to build Bob, or
  unexpected problems might occur.

  If Bob is installed by the administrator of your system, it is safe to
  consider it uses the default python interpreter. In this case, the above 3
  command lines should work as expected. If you have Bob installed somewhere
  else on a private directory, edit the file ``localbob.cfg`` and use that
  with ``buildout`` instead::

    $ python boostrap.py
    $ # edit localbob.cfg
    $ ./bin/buildout -c localbob.cfg

Usage
-----

Please refer to the documentation inside the ``doc`` directory of this package
for further instructions on the functionality available.

Reference
---------

If you need to cite this work, please use the following::

   @INPROCEEDINGS{Chingovska_BIOSIG_2012,
   author = {Chingovska, Ivana and Anjos, Andr{\'{e}} and Marcel, S{\'{e}}bastien},
   keywords = {Attack, Counter-Measures, Counter-Spoofing, Face Recognition, Liveness Detection, Replay, Spoofing},
   month = sep,
   title = {On the Effectiveness of Local Binary Patterns in Face Anti-spoofing},
   journal = {IEEE BIOSIG 2012},
   year = {2012},
   }

