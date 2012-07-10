#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Thu Jan 19 15:24:14 CET 2012

"""This script makes a histogram models for the real accesses videos in REPLAY-ATTACK by averaging the LBP histograms of each real access video. The output is an hdf5 file with the computed model histograms. The procedure is described in the paper: "On the Effectiveness of Local Binary patterns in Face Anti-spoofing" - Chingovska, Anjos & Marcel; BIOSIG 2012
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
  OUTPUT_DIR = os.path.join(basedir, 'res')

  protocols = bob.db.replay.Database().protocols()
  
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the histogram features of all the videos')
  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results (models).')
  parser.add_argument('-p', '--protocol', metavar='PROTOCOL', type=str, dest="protocol", default='grandtest', help='The protocol type may be specified instead of the the id switch to subselect a smaller number of files to operate on', choices=protocols)   
  
  args = parser.parse_args()
  if not os.path.exists(args.inputdir):
    parser.error("input directory does not exist")
  
  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)
    
  print "Output directory set to \"%s\"" % args.outputdir
  print "Loading input files..."

  # loading the input files
  db = bob.db.replay.Database()

  process_train_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='train', cls='real')

  # create the full datasets from the file data
  train_real = create_full_dataset(process_train_real);
  
  print "Creating the model..."

  model_hist_real = [sum(train_real[:,i]) for i in range(0, train_real.shape[1])] # sum the histograms of the real access videos
  
  model_hist_real = [i / train_real.shape[0] for i in model_hist_real]  # average the model histogram for the real access videos

  print "Saving the model histograms..."
  histmodelsfile = bob.io.HDF5File(os.path.join(args.outputdir, 'histmodelsfile.hdf5'),'w')
  histmodelsfile.append('model_hist_real', numpy.array(model_hist_real))

  del histmodelsfile
 
if __name__ == '__main__':
  main()
