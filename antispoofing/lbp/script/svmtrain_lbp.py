#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Wed Mar 28 14:26:11 CEST 2012

"""This script can makes an SVM classification of data into two categories: real accesses and spoofing attacks. There is an option for normalizing between [-1, 1] and dimensionality reduction of the data prior to the SVM classification.
The probabilities obtained with the SVM are considered as scores for the data. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and HTER for the test and development set are calculated. The script outputs a text file with the performance results.
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
  OUTPUT_DIR = os.path.join(basedir, 'res')

  protocols = bob.db.replay.Database().protocols()

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the scores to be loaded')
  parser.add_argument('-d', '--output-dir', metavar='DIR', type=str, dest='outputdir', default=OUTPUT_DIR, help='Base directory that will be used to save the results.')
  parser.add_argument('-n', '--normalize', action='store_true', dest='normalize', default=False, help='If True, will do zero mean unit variance normalization on the data before creating the LDA machine')
  parser.add_argument('-r', '--pca_reduction', action='store_true', dest='pca_reduction', default=False, help='If set, PCA dimensionality reduction will be performed to the data before doing LDA')
  parser.add_argument('-e', '--energy', type=str, dest="energy", default='0.99', help='The energy which needs to be preserved after the dimensionality reduction if PCA is performed prior to LDA')
  parser.add_argument('-p', '--protocol', metavar='PROTOCOL', type=str, dest="protocol", default='grandtest', help='The protocol type may be specified instead of the the id switch to subselect a smaller number of files to operate on', choices=protocols) 

  from .. import ml
  from ..ml import pca, norm

  args = parser.parse_args()
  if not os.path.exists(args.inputdir):
    parser.error("input directory does not exist")

  if not os.path.exists(args.outputdir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(args.outputdir)

  energy = float(args.energy)

  print "Loading input files..."
  # loading the input files
  db = bob.db.replay.Database()

  process_train_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='train', cls='real')
  process_train_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='train', cls='attack')
  process_devel_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='devel', cls='real')
  process_devel_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='devel', cls='attack')
  process_test_real = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='test', cls='real')
  process_test_attack = db.files(directory=args.inputdir, extension='.hdf5', protocol=args.protocol, groups='test', cls='attack')

  # create the full datasets from the file data
  train_real = create_full_dataset(process_train_real); train_attack = create_full_dataset(process_train_attack); 
  devel_real = create_full_dataset(process_devel_real); devel_attack = create_full_dataset(process_devel_attack); 
  test_real = create_full_dataset(process_test_real); test_attack = create_full_dataset(process_test_attack); 

  if args.normalize:  # normalization in the range [-1, 1] (recommended by LIBSVM)
    train_data = numpy.concatenate((train_real, train_attack), axis=0) 
    mins, maxs = norm.calc_min_max(train_data)
    train_real = norm.norm_range(train_real, mins, maxs, -1, 1); train_attack = norm.norm_range(train_attack, mins, maxs, -1, 1)
    devel_real = norm.norm_range(devel_real, mins, maxs, -1, 1); devel_attack = norm.norm_range(devel_attack, mins, maxs, -1, 1)
    test_real = norm.norm_range(test_real, mins, maxs, -1, 1); test_attack = norm.norm_range(test_attack, mins, maxs, -1, 1)
  
  if args.pca_reduction: # PCA dimensionality reduction of the data
    train = bob.io.Arrayset() # preparing the train data for PCA (putting them altogether into bob.io.Arrayset)
    train.extend(train_real)
    train.extend(train_attack)
    pca_machine = pca.make_pca(train, energy, False) # performing PCA
    train_real = pca.pcareduce(pca_machine, train_real); train_attack = pca.pcareduce(pca_machine, train_attack)
    devel_real = pca.pcareduce(pca_machine, devel_real); devel_attack = pca.pcareduce(pca_machine, devel_attack)
    test_real = pca.pcareduce(pca_machine, test_real); test_attack = pca.pcareduce(pca_machine, test_attack)

  print "Training SVM machine..."
  svm_trainer = bob.trainer.SVMTrainer()
  svm_trainer.probability = True
  svm_machine = svm_trainer.train([train_real, train_attack])
  svm_machine.save(os.path.join(args.outputdir, 'svm_machine.txt'))

  def svm_predict(svm_machine, data):
    labels = [svm_machine.predict_class_and_scores(x)[1][0] for x in data]
    return labels
  
  print "Computing devel and test scores..."
  devel_real_out = svm_predict(svm_machine, devel_real);
  devel_attack_out = svm_predict(svm_machine, devel_attack);
  test_real_out = svm_predict(svm_machine, test_real);
  test_attack_out = svm_predict(svm_machine, test_attack);

 # it is expected that the scores of the real accesses are always higher then the scores of the attacks. Therefore, a check is first made, if the average of the scores of real accesses is smaller then the average of the scores of the attacks, all the scores are inverted by multiplying with -1.
  if numpy.mean(devel_real_out) < numpy.mean(devel_attack_out):
    devel_real_out = devel_real_out * -1; devel_attack_out = devel_attack_out * -1
    test_real_out = test_real_out * -1; test_attack_out = test_attack_out * -1
    
  # calculation of the error rates
  thres = bob.measure.eer_threshold(devel_attack_out, devel_real_out)
  dev_far, dev_frr = bob.measure.farfrr(devel_attack_out, devel_real_out, thres)
  test_far, test_frr = bob.measure.farfrr(test_attack_out, test_real_out, thres)
  
  tbl = []
  tbl.append(" ")
  if args.pca_reduction:
    tbl.append("EER @devel - (energy kept after PCA = %.2f" % (energy))
  tbl.append(" threshold: %.4f" % thres)
  tbl.append(" dev:  FAR %.2f%% (%d / %d) | FRR %.2f%% (%d / %d) | HTER %.2f%% " % \
      (100*dev_far, int(round(dev_far*len(devel_attack))), len(devel_attack), 
       100*dev_frr, int(round(dev_frr*len(devel_real))), len(devel_real),
       50*(dev_far+dev_frr)))
  tbl.append(" test: FAR %.2f%% (%d / %d) | FRR %.2f%% (%d / %d) | HTER %.2f%% " % \
      (100*test_far, int(round(test_far*len(test_attack))), len(test_attack),
       100*test_frr, int(round(test_frr*len(test_real))), len(test_real),
       50*(test_far+test_frr)))
  txt = ''.join([k+'\n' for k in tbl])

  print txt

  # write the results to a file 
  tf = open(os.path.join(args.outputdir, 'perf_table.txt'), 'w')
  tf.write(txt)
 
if __name__ == '__main__':
  main()
