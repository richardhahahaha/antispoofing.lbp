#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Thu Jan 19 15:56:55 CET 2012

"""This script calculates the chi2 difference between a model histogram and the data histograms, assigning scores to the data according to this. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and HTER for the test and development set are calculated. The script outputs a text file with the performance results.
The details about the procedure are described in the paper: "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
"""

import os, sys
import argparse
import bob
import numpy

def create_full_dataset(files):
  """Creates a full dataset matrix out of all the specified files"""
  dataset = None
  for key, filename in files.items():
    filename = os.path.expanduser(filename)
    fvs = bob.io.load(filename)
    if dataset is None:
      dataset = fvs
    else:
      dataset = numpy.append(dataset, fvs, axis = 0)
  return dataset

def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  INPUT_MODEL_DIR = os.path.join(basedir, 'res')
  OUTPUT_DIR = os.path.join(basedir, 'res')
  
  protocols = bob.db.replay.Database().protocols()

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')
  parser.add_argument('-m', '--input-modeldir', metavar='DIR', type=str, dest='inputmodeldir', default=INPUT_MODEL_DIR, help='Base directory containing the histogram models to be loaded')
  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-p', '--protocol', metavar='PROTOCOL', type=str, dest="protocol", default='grandtest', help='The protocol type may be specified instead of the the id switch to subselect a smaller number of files to operate on', choices=protocols) 

  from .. import ml
  from .. import spoof
  from ..ml import perf
  from ..spoof import chi2

  args = parser.parse_args()
  if not os.path.exists(args.inputdir) or not os.path.exists(args.inputmodeldir):
    parser.error("input directory does not exist")

  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)
    
  print "Output directory set to \"%s\"" % args.outputdir
  print "Loading input files..."

  # loading the input files (all the feature vectors of all the files in different subdatasets)
  db = bob.db.replay.Database()

  process_devel_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='devel', cls='real')
  process_devel_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='devel', cls='attack')
  process_test_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='test', cls='real')
  process_test_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='test', cls='attack')

  # create the full datasets from the file data
  devel_real = create_full_dataset(process_devel_real); devel_attack = create_full_dataset(process_devel_attack); 
  test_real = create_full_dataset(process_test_real); test_attack = create_full_dataset(process_test_attack); 

  print "Loading the models..."
  # loading the histogram models
  histmodelsfile = bob.io.HDF5File(os.path.join(args.inputmodeldir, 'histmodelsfile.hdf5'),'r')
  model_hist_real = histmodelsfile.read('model_hist_real')
  del histmodelsfile
    
  model_hist_real = model_hist_real[0,:]

  print "Calculating the Chi-2 differences..."
  # calculating the comparison scores with chi2 distribution for each protocol subset   
  sc_devel_realmodel = chi2.cmphistbinschimod(model_hist_real, (devel_real, devel_attack))
  sc_test_realmodel = chi2.cmphistbinschimod(model_hist_real, (test_real, test_attack))


  print "Saving the results in a file"
  # It is expected that the positives always have larger scores. Therefore, it is necessary to "invert" the scores by multiplying them by -1 (the chi-square test gives smaller scores to the data from the similar distribution)
  sc_devel_realmodel = (sc_devel_realmodel[0] * -1, sc_devel_realmodel[1] * -1)
  sc_test_realmodel = (sc_test_realmodel[0] * -1, sc_test_realmodel[1] * -1)

  perftable, eer_thres, mhter_thres = perf.performance_table(sc_test_realmodel, sc_devel_realmodel, "CHI-2 comparison, RESULTS")
  tf = open(os.path.join(args.outputdir, 'perf_table.txt'), 'w')
  tf.write(perftable)
  tf.close()  

if __name__ == '__main__':
  main()
