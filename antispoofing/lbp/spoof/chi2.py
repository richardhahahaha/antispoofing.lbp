#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Fri Jan 20 11:07:38 CET 2012

"""Support methods to perform Chi-square comparison between two histogram models
"""

import numpy

def cmphistbinschimod(model, data):
  """ Calculates the chi-square distribution scores of similarity of the data according to the model, but using the modified chi-square difference (for the formula of the Chi-2 difference see paper "Face Recognition for Local Binary Patterns" - Ahonen, Hadid, Pietikainen). The returned score for each sample in the data is the probablility that that sample comes from the model distribution. 
   
      Keyword parameters:

      model
        The model distribution
      data
        A tuple whose first element is a 2D array of the real access histogram data, and the second element is a 2D array of the attack histogram data
   
      Returns:
    
        A tuple whose first element is a 2D column array of the scores of the real access data and the second a 2D column array of the scores of the attack data
  """
  data_real = data[0]
  data_attack = data[1]
  scores_real = numpy.ndarray((data_real.shape[0],1), 'float64') # initialize array for scores of the real data
  scores_attack =  numpy.ndarray((data_attack.shape[0],1), 'float64') # initialize array for scores of the attack data

  for k in range(0, data_real.shape[0]): # calc scores for the real data
    tmp = numpy.square(data_real[k,:] - model)
    s = sum(numpy.nan_to_num(tmp / (model + data_real[k,:])))
    scores_real[k,0] = s

  for k in range(0, data_attack.shape[0]): # calc scores for the attack data
    tmp = numpy.square(data_attack[k,:] - model)
    s = sum(numpy.nan_to_num(tmp / (model + data_attack[k,:])))
    scores_attack[k,0] = s
  
  return (scores_real, scores_attack)
