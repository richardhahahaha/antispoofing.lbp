#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 19 Sep 2011 15:01:44 CEST 

"""LDA training for the anti-spoofing library
"""

import bob
import numpy

def make_lda(train, verbose=False):
  """Creates a new linear machine and train it using LDA.

  Keyword Parameters:

  train
    An iterable (tuple or list) containing two arraysets: the first contains
    the real accesses and the second contains the attacks.

  verbose
    Makes the training more verbose
  """

  # checking the type of the data provided for reduction
  if type(train[0]) == numpy.ndarray:
    # putting the numpy.ndarray data into Arrayset
    t0 = bob.io.Arrayset()
    t1 = bob.io.Arrayset()
    t0.extend(train[0])
    t1.extend(train[1])
    train_array = (t0, t1)
  else:
    train_array = train

  T = bob.trainer.FisherLDATrainer()
  machine, eig_vals = T.train(train_array)
  return machine

def get_scores(machine, data):
  """Gets the scores for the data"""

  # checking the type of the data provided for reduction
  if type(data) == numpy.ndarray:
    # putting the numpy.ndarray data into Arrayset
    dataarray = bob.io.Arrayset()
    dataarray.extend(data)
    return numpy.vstack(dataarray.foreach(machine))[:,0]  #the new vectors with reduced dimensionality

  return numpy.vstack(data.foreach(machine))[:,0]
