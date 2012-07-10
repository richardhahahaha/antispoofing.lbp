#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Fri Jan 27 11:13:59 CET 2012

"Utility functions for performing PCA dimensionality reduction of data "

import bob
import numpy

def make_pca(data, perc, norm=False):
  """ Creates a new LinearMachine for PCA reduction of data, using the training data given as argument. Returns a bob.machine.LinearMachine containing a numpy.ndarray of the most important eigenvectors.

  Keyword parameters:

  data
    numpy.ndarray or bob.io.Arrayset containing the training data which will be used to calculate the PCA parameters

  perc
    the percentage of energy which should be conserved when reducing the dimensions

  norm
    if set to True, unit-variance normalization will be done to the data prior to reduction (zero mean is done by default anyway)
"""

  # checking the type of the data provided for reduction
  if type(data) == numpy.ndarray:
    # putting the numpy.ndarray data into Arrayset
    dataarray = bob.io.Arrayset()
    dataarray.extend(data)
  else: dataarray = data

  T = bob.trainer.SVDPCATrainer(zscore_convert=norm) # zero-mean, unit-variance will be performed prior to reduction
  params = T.train(dataarray) # params contain a tuple (eigenvecetors, eigenvalues) sorted in descending order

  eigvalues = params[1]
  
  # calculating the cumulative energy of the eigenvalues
  cumEnergy = [sum(eigvalues[0:eigvalues.size-i]) / sum(eigvalues) for i in range(0, eigvalues.size+1)]
  
  # calculating the number of eigenvalues to keep the required energy
  numeigvalues = eigvalues.size
  for i in range(0, len(cumEnergy)-1):
    if cumEnergy[i] < perc:
      numeigvalues = len(cumEnergy) - i
      break
    
  # recalculating the shape of the LinearMachine
  oldshape = params[0].shape
  params[0].resize(oldshape[0], numeigvalues) # the second parameter gives the number of kept eigenvectors/eigenvalues

  return params[0]


def pcareduce(machine, data):
  """ Reduces the dimension of the data, using the given bob.machine.LinearMachine (projects each of the data feature vector in the lower dimensional space). Returns numpy.ndarray of the feature vectors with reduced dimensionality. The accepted input data can be in numpy.ndarray format or bob.io.Arrayset format.

  Keyword parameters:

  machine
    bob.machine.LinearMachine
  data
    numpy.ndarray or bob.io.Arrayset() containing the data which need to be reduced

  perc
    the percentage of energy which should be conserved when reducing the dimensions
  """
  
  # checking the type of the data provided for reduction
  if type(data) == numpy.ndarray:
    # putting the numpy.ndarray data into Arrayset
    dataarray = bob.io.Arrayset()
    dataarray.extend(data)
    return numpy.vstack(dataarray.foreach(machine))  #the new vectors with reduced dimensionality
 
  # if the data is bob.io.Arrayset
  return numpy.vstack(data.foreach(machine))   #the new vectors with reduced dimensionality
