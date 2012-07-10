#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon Mar 26 12:07:09 CEST 2012

import bob
import numpy
import math

""" Utility functions for column-wise normalization of data
"""

def calc_mean(c0, c1):
  """ Calculates the mean of the data. The input is in bob.io.Arrayset format"""
  return (numpy.mean(c0, 0) + numpy.mean(c1, 0)) / 2.

def calc_std(c0, c1):
  """ Calculates the variance of the data. The input is in bob.io.Arrayset format"""
  prop = float(len(c0)) / float(len(c1))
  if prop < 1: 
    p0 = int(math.ceil(1/prop))
    p1 = 1
  else:
    p0 = 1
    p1 = int(math.ceil(prop))
  return numpy.std(numpy.vstack(p0*[c0] + p1*[c1]), 0)
  
def calc_mean_std(c0, c1):
  """ Calculates both the mean of the data. The input is in bob.io.Arrayset format"""
  return calc_mean(c0, c1), calc_std(c0, c1)

def calc_mean_std_clip(c0, c1):
  """ Calculates both the mean of the data. The input is in bob.io.Arrayset format"""
  x0 = numpy.clip(c0, 0., 300.)
  x1 = numpy.clip(c1, 0., 300.)
  return calc_mean(x0, x1), calc_std(x0, x1)

def calc_bounds(c0, c1):
  mn = numpy.min(numpy.vstack([numpy.min(c0, 0), numpy.min(c1, 0)]), 0)
  mx = numpy.max(numpy.vstack([numpy.max(c0, 0), numpy.max(c1, 0)]), 0)
  div = (mx - mn) / 2.0
  sub = (div + mn)
  return (sub, div)

def zeromean_unitvar_norm(data, mean, std):
  """ Normalized the data with zero mean and unit variance. The data is in bob.io.Arrayset format. Mean and variance are in numpy.ndarray format"""
  return bob.io.Arrayset( (data-mean)/std )

def zeromean_unitvar_norm_clip(data, mean, std):
  """ Normalized the data with zero mean and unit variance. The data is in bob.io.Arrayset format. Mean and variance are in numpy.ndarray format"""
  return bob.io.Arrayset( (numpy.clip(data, 0., 300.)-mean)/std )

def calc_min_max(data):
  """Calculation of the minimum and maximum of each feature in a dataset"""
  array_min = [min(data[:,i]) for i in range(0, data.shape[1])]
  array_max = [max(data[:,i]) for i in range(0, data.shape[1])]
  return numpy.array(array_min), numpy.array(array_max)

def norm_range(data, mins, maxs, lowbound, highbound):
  """ Normalizing the data with range normalization between lowbound and highbound
  
  Keyword parameters:
  
  data
    the data to be normalized, numpy.ndarray, each row is a sample

  mins, maxs
    arrays of minimum and maximum values that each feature can take

  lowbound, highbound
    the bounds of the normalization
"""
  denom = maxs - mins
  diff = highbound - lowbound
  addit = numpy.ndarray([data.shape[0],1])
  addit.fill(lowbound)
  for i in range(data.shape[0]): # for each feature vector
    data[i] = diff * (data[i] - mins) / denom + lowbound
    data[i][numpy.isnan(data[i])] = (lowbound + highbound) / 2
  return data
