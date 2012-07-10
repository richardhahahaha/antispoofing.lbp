#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 17 Aug 11:42:09 2011 

"""A few utilities to plot and dump results.
"""

import os
import bob
import numpy
import re

def pyplot_axis_fontsize(ax, size):
  """Sets the font size on axis labels"""

  for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(size)
  for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(size)

def score_distribution_plot(test, devel, train, epochs, bins, eer_thres,
    mhter_thres):
  """Plots the score distributions in 3 different subplots"""

  import matplotlib.pyplot as mpl
  histoargs = {'bins': bins, 'alpha': 0.8, 'histtype': 'step', 'range': (-1,1)} 
  lineargs = {'alpha': 0.5}
  axis_fontsize = 8

  # 3 plots (same page) with the tree sets
  mpl.subplot(3,1,1)
  mpl.hist(test[0][:,0], label='Real Accesses', color='g', **histoargs)
  mpl.hist(test[1][:,0], label='Attacks', color='b', **histoargs)
  xmax, xmin, ymax, ymin = mpl.axis()
  mpl.vlines(eer_thres, ymin, ymax, color='red', label='EER', 
      linestyles='solid', **lineargs)
  mpl.vlines(mhter_thres, ymin, ymax, color='magenta', 
      linestyles='dashed', label='Min.HTER', **lineargs)
  mpl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=4, mode="expand", borderaxespad=0.)
  mpl.grid(True, alpha=0.5)
  mpl.ylabel("Test set")
  axis = mpl.gca()
  axis.yaxis.set_label_position('right')
  pyplot_axis_fontsize(axis, axis_fontsize)
  mpl.subplot(3,1,2)
  mpl.hist(devel[0][:,0], color='g', **histoargs)
  mpl.hist(devel[1][:,0], color='b', **histoargs)
  xmax, xmin, ymax, ymin = mpl.axis()
  mpl.vlines(eer_thres, ymin, ymax, color='red', linestyles='solid',
      label='EER', **lineargs)
  mpl.vlines(mhter_thres, ymin, ymax, color='magenta', linestyles='dashed',
      label='Min.HTER', **lineargs)
  mpl.grid(True, alpha=0.5)
  mpl.ylabel("Development set")
  axis = mpl.gca()
  axis.yaxis.set_label_position('right')
  pyplot_axis_fontsize(axis, axis_fontsize)
  mpl.subplot(3,1,3)
  mpl.hist(train[0][:,0], color='g', **histoargs)
  mpl.hist(train[1][:,0], color='b', **histoargs)
  xmax, xmin, ymax, ymin = mpl.axis()
  mpl.vlines(eer_thres, ymin, ymax, color='red', linestyles='solid', 
      label='EER', **lineargs)
  mpl.vlines(mhter_thres, ymin, ymax, color='magenta', linestyles='dashed',
      label='Min.HTER', **lineargs)
  mpl.grid(True, alpha=0.5)
  mpl.ylabel("Training set")
  mpl.xlabel("Score distribution after training (%d steps)" % epochs) 
  axis = mpl.gca()
  axis.yaxis.set_label_position('right')
  pyplot_axis_fontsize(axis, axis_fontsize)


def perf_hter(test_scores, devel_scores, threshold_func):
  """Computes a performance table and returns the HTER for the test and development set, as well as a formatted text with the results and the value of the threshold obtained for the given threshold function
     Keyword parameters:
       test_scores - the scores of the samples in the test set
       devel_scores - the scores of the samples in the development set
       threshold function - the type of threshold
  """ 
   
  from bob.measure import farfrr

  devel_attack_scores = devel_scores[1][:,0]
  devel_real_scores = devel_scores[0][:,0]
  test_attack_scores = test_scores[1][:,0]
  test_real_scores = test_scores[0][:,0]

  devel_real = devel_real_scores.shape[0]
  devel_attack = devel_attack_scores.shape[0]
  test_real = test_real_scores.shape[0]
  test_attack = test_attack_scores.shape[0]

  thres = threshold_func(devel_attack_scores, devel_real_scores)
  devel_far, devel_frr = farfrr(devel_attack_scores, devel_real_scores, thres)
  test_far, test_frr = farfrr(test_attack_scores, test_real_scores, thres)
  devel_hter = 50 * (devel_far + devel_frr)
  test_hter = 50 * (test_far + test_frr)
  devel_text = " d: FAR %.2f%% / FRR %.2f%% / HTER %.2f%% " % (100*devel_far, 100*devel_frr, devel_hter)
  test_text = " t: FAR %.2f%% / FRR %.2f%% / HTER %.2f%% " % (100*test_far, 100*test_frr, test_hter)
  return (test_hter, devel_hter), (test_text, devel_text), thres

def perf_hter_thorough(test_scores, devel_scores, threshold_func):
  """Computes a performance table and returns the HTER for the test and development set, as well as a formatted text with the results and the value of the threshold obtained for the given threshold function
     Keyword parameters:
       test_scores - the scores of the samples in the test set (tuple)
       devel_scores - the scores of the samples in the development set (tuple)
       threshold function - the type of threshold
  """ 
   
  from bob.measure import farfrr

  devel_attack_scores = devel_scores[1]
  devel_real_scores = devel_scores[0]
  test_attack_scores = test_scores[1]
  test_real_scores = test_scores[0]
  
  devel_attack_scores = devel_attack_scores.reshape([len(devel_attack_scores)]) # all the scores whould be arrays with shape (n,)
  devel_real_scores = devel_real_scores.reshape([len(devel_real_scores)])
  test_attack_scores = test_attack_scores.reshape([len(test_attack_scores)])
  test_real_scores = test_real_scores.reshape([len(test_real_scores)])

  thres = threshold_func(devel_attack_scores, devel_real_scores)
  devel_far, devel_frr = farfrr(devel_attack_scores, devel_real_scores, thres)
  test_far, test_frr = farfrr(test_attack_scores, test_real_scores, thres)
  return (devel_far, devel_frr), (test_far, test_frr)

def performance_table(test, devel, title):
  """Returns a string containing the performance table"""

  def pline(group, far, attack_count, frr, real_count):
    fmtstr = " %s: FAR %.2f%% (%d / %d) / FRR %.2f%% (%d / %d) / HTER %.2f%%"
    return fmtstr % (group,
        100 * far, int(round(far*attack_count)), attack_count, 
        100 * frr, int(round(frr*real_count)), real_count, 
        50 * (far + frr))

  def perf(devel_scores, test_scores, threshold_func):
  
    from bob.measure import farfrr

    devel_attack_scores = devel_scores[1][:,0]
    devel_real_scores = devel_scores[0][:,0]
    test_attack_scores = test_scores[1][:,0]
    test_real_scores = test_scores[0][:,0]

    devel_real = devel_real_scores.shape[0]
    devel_attack = devel_attack_scores.shape[0]
    test_real = test_real_scores.shape[0]
    test_attack = test_attack_scores.shape[0]

    thres = threshold_func(devel_attack_scores, devel_real_scores)
    devel_far, devel_frr = farfrr(devel_attack_scores, devel_real_scores, thres)
    test_far, test_frr = farfrr(test_attack_scores, test_real_scores, thres)

    retval = []
    retval.append(" threshold: %.4f" % thres)
    retval.append(pline("dev ", devel_far, devel_attack, devel_frr, devel_real))
    retval.append(pline("test", test_far, test_attack, test_frr, test_real))

    return retval, thres

  retval = []
  retval.append(title)
  retval.append("")
  retval.append("EER @ devel")
  eer_table, eer_thres = perf(devel, test, bob.measure.eer_threshold)
  retval.extend(eer_table)
  retval.append("")
  retval.append("Mininum HTER @ devel")
  mhter_table, mhter_thres = perf(devel, test, bob.measure.min_hter_threshold)
  retval.extend(mhter_table)
  retval.append("")

  return ''.join([k+'\n' for k in retval]), eer_thres, mhter_thres

def roc(test, devel, train, npoints, eer_thres, mhter_thres):
  """Plots the ROC curve using Matplotlib"""

  import matplotlib.pyplot as mpl
  import matplotlib.patches as mpp

  dev_neg = devel[1][:,0]
  dev_pos = devel[0][:,0]
  test_neg = test[1][:,0]
  test_pos = test[0][:,0]
  train_neg = train[1][:,0]
  train_pos = train[0][:,0]

  bob.measure.plot.roc(train_neg, train_pos, npoints, color=(0.3,0.3,0.3),
      linestyle='--', dashes=(6,2), alpha=0.5, label='training')
  bob.measure.plot.roc(dev_neg, dev_pos, npoints, color=(0.3,0.3,0.3), 
      linestyle='--', dashes=(6,2), label='development')
  bob.measure.plot.roc(test_neg, test_pos, npoints, color=(0,0,0),
      linestyle='-', label='test')

  eer_far, eer_frr = bob.measure.farfrr(test_neg, test_pos, eer_thres)
  mhter_far, mhter_frr = bob.measure.farfrr(test_neg, test_pos, mhter_thres)

  xmax = min(100,2*100*eer_frr)
  if xmax < 5: xmax = 5
  ymax = min(100,2*100*eer_far)
  if ymax < 5: ymax = 5
  
  mpl.axis([0,xmax,0,ymax])

  # roundness impression for the ellipse
  xratio = float(xmax)/ymax
  radius = 0.7

  # for the test set line
  ax = mpl.gca()
  exy = (100*eer_frr, 100*eer_far)
  ax.add_patch(mpp.Ellipse(exy, radius*xratio, radius, color='r', alpha=0.7,
    label='EER'))
  exy = (100*mhter_frr, 100*mhter_far)
  ax.add_patch(mpp.Ellipse(exy, radius*xratio, radius, color='m', alpha=0.7,
    label='Min.HTER'))

  # for the development set line
  eer_far, eer_frr = bob.measure.farfrr(dev_neg, dev_pos, eer_thres)
  mhter_far, mhter_frr = bob.measure.farfrr(dev_neg, dev_pos, mhter_thres)
  exy = (100*eer_frr, 100*eer_far)
  ax.add_patch(mpp.Ellipse(exy, radius*xratio, radius, color='r', alpha=0.2,
    hatch='/'))
  exy = (100*mhter_frr, 100*mhter_far)
  ax.add_patch(mpp.Ellipse(exy, radius*xratio, radius, color='m', alpha=0.2,
    hatch='/'))
  
  mpl.title("ROC Curve")
  mpl.xlabel('FRR (%)')
  mpl.ylabel('FAR (%)')
  mpl.grid(True, alpha=0.3)
  mpl.legend()

def det(test, devel, train, npoints, eer_thres, mhter_thres):
  """Plots the DET curve using Matplotlib"""

  import matplotlib.pyplot as mpl
  import matplotlib.patches as mpp

  dev_neg = devel[1][:,0]
  dev_pos = devel[0][:,0]
  test_neg = test[1][:,0]
  test_pos = test[0][:,0]
  train_neg = train[1][:,0]
  train_pos = train[0][:,0]

  bob.measure.plot.det(train_neg, train_pos, npoints, color=(0.3,0.3,0.3), 
      linestyle='--', dashes=(6,2), alpha=0.5, label='training')
  bob.measure.plot.det(dev_neg, dev_pos, npoints, color=(0.3,0.3,0.3), 
      linestyle='--', dashes=(6,2), label='development')
  bob.measure.plot.det(test_neg, test_pos, npoints, color=(0,0,0),
      linestyle='-', label='test')

  eer_far, eer_frr = bob.measure.farfrr(test_neg, test_pos, eer_thres)
  mhter_far, mhter_frr = bob.measure.farfrr(test_neg, test_pos, mhter_thres)
  
  xmax = min(99.99, 4*100*eer_frr)
  if xmax < 5.: xmax = 5. 
  ymax = min(99.99, 4*100*eer_far)
  if ymax < 5.: ymax = 5.
  
  bob.measure.plot.det_axis([0.01, xmax, 0.01, ymax])

  # roundness impression for the ellipse
  xratio = xmax/ymax
  radius = 0.07

  # for the test set line
  ax = mpl.gca()
  exy = [bob.measure.ppndf(k) for k in (eer_frr, eer_far)]
  ax.add_patch(mpp.Ellipse(exy, radius*xratio, radius, color='r', alpha=0.7,
    label='EER'))
  exy = [bob.measure.ppndf(k) for k in (mhter_frr, mhter_far)]
  ax.add_patch(mpp.Ellipse(exy, radius*xratio, radius, color='m', alpha=0.7,
    label='Min.HTER'))

  # for the development set line
  eer_far, eer_frr = bob.measure.farfrr(dev_neg, dev_pos, eer_thres)
  mhter_far, mhter_frr = bob.measure.farfrr(dev_neg, dev_pos, mhter_thres)
  exy = [bob.measure.ppndf(k) for k in (eer_frr, eer_far)]
  ax.add_patch(mpp.Ellipse(exy, radius*xratio, radius, color='r', alpha=0.2,
    hatch='/'))
  exy = [bob.measure.ppndf(k) for k in (mhter_frr, mhter_far)]
  ax.add_patch(mpp.Ellipse(exy, radius*xratio, radius, color='m', alpha=0.2,
    hatch='/'))
  
  mpl.title("DET Curve")
  mpl.xlabel('FRR (%)')
  mpl.ylabel('FAR (%)')
  mpl.grid(True, alpha=0.3)
  mpl.legend()

def epc(test, devel, npoints):
  """Plots the EPC curve using Matplotlib"""
  
  import matplotlib.pyplot as mpl

  dev_neg = devel[1][:,0]
  dev_pos = devel[0][:,0]
  test_neg = test[1][:,0]
  test_pos = test[0][:,0]

  bob.measure.plot.epc(dev_neg, dev_pos, test_neg, test_pos, npoints, 
      color=(0,0,0), linestyle='-')
  mpl.title('EPC Curve')
  mpl.xlabel('Cost')
  mpl.ylabel('Min. HTER (%)')
  mpl.grid(True, alpha=0.3)

def plot_rmse_evolution(data):
  """Performance evolution during training"""

  import matplotlib.pyplot as mpl

  mpl.plot(data['epoch'], data['real-train-rmse'], color='green',
    linestyle='--', dashes=(6,2), alpha=0.5, label='Real Access (train)')
  mpl.plot(data['epoch'], data['attack-train-rmse'], color='blue',
    linestyle='--', dashes=(6,2), alpha=0.5, label='Attack (train)')
  train = [0.5*sum(k) for k in zip(data['real-train-rmse'],
    data['attack-train-rmse'])]
  mpl.plot(data['epoch'], train, color='black',
    linestyle='--', dashes=(6,2), label='Total (train)')

  mpl.plot(data['epoch'], data['real-devel-rmse'], color='green',
    alpha=0.5, label='Real Access (devel)')
  mpl.plot(data['epoch'], data['attack-devel-rmse'], color='blue',
    alpha=0.5, label='Attack (devel)')
  devel = [0.5*sum(k) for k in zip(data['real-devel-rmse'],
    data['attack-devel-rmse'])]
  mpl.plot(data['epoch'], devel, color='black', label='Total (devel)')
  
  mpl.title('RMSE Evolution')
  mpl.xlabel('Training steps')
  mpl.ylabel('RMSE')
  mpl.grid(True, alpha=0.3)
  mpl.legend()

  # Reduce the size of the legend text
  leg = mpl.gca().get_legend()
  ltext  = leg.get_texts()
  mpl.setp(ltext, fontsize='small')

def plot_eer_evolution(data):
  """Performance evolution during training"""

  import matplotlib.pyplot as mpl

  train = [50*sum(k) for k in zip(data['train-frr'], data['train-far'])]
  mpl.plot(data['epoch'], train, color='black', alpha=0.6,
    linestyle='--', dashes=(6,2), label='EER (train)')

  devel = [50*sum(k) for k in zip(data['devel-frr'], data['devel-far'])]
  mpl.plot(data['epoch'], devel, color='black', label='EER (devel)')
  
  mpl.title('EER Evolution (threshold from training set)')
  mpl.xlabel('Training steps')
  mpl.ylabel('Equal Error Rate')
  mpl.grid(True, alpha=0.3)
  mpl.legend()

  # Reduce the size of the legend text
  leg = mpl.gca().get_legend()
  ltext  = leg.get_texts()
  mpl.setp(ltext, fontsize='small')

def evaluate_relevance(test, devel, train, machine):
  """Evaluates the relevance of each component"""

  import matplotlib.pyplot as mpl

  test_relevance = bob.measure.relevance(numpy.vstack(test), 
      machine)
  test_relevance = [test_relevance[k] for k in range(test_relevance.shape[0])]
  devel_relevance = bob.measure.relevance(numpy.vstack(devel), 
      machine)
  devel_relevance = [devel_relevance[k] for k in range(devel_relevance.shape[0])]
  train_relevance = bob.measure.relevance(numpy.vstack(train), 
      machine)
  train_relevance = [train_relevance[k] for k in range(train_relevance.shape[0])]

  data_width = len(test_relevance)
  spacing = 0.1
  width = (1.0-(2*spacing))/3.0
  train_bottom = [k+spacing for k in range(data_width)]
  devel_bottom = [k+width for k in train_bottom]
  test_bottom = [k+width for k in devel_bottom]

  mpl.barh(test_bottom, test_relevance, width, label='Test', color='black')
  mpl.barh(devel_bottom, devel_relevance, width, label='Development',
      color=(0.4, 0.4, 0.4))
  mpl.barh(train_bottom, train_relevance, width, label='Training',
      color=(0.9, 0.9, 0.9))

  # labels and other details
  ax = mpl.gca()
  ax.set_yticks([k+spacing+1.5*width for k in range(data_width)])
  ax.set_yticklabels(range(1,data_width+1))
  mpl.title('Feature Relevance')
  mpl.ylabel('Components')
  mpl.xlabel('Relevance')
  mpl.legend()
  mpl.grid(True, alpha=0.3)

def get_hter(machine, datadir, mhter=False, verbose=False):
  """Returns the HTER on the test and development sets given the machine and
  data directories.  If the flag 'mhter' is set to True, then calculate the
  test set HTER using the threshold found by minimizing the HTER on the
  development set, otherwise, use the threshold at the EER on the development
  set."""
  
  def loader(group, cls, inputdir, verbose):
    filename = os.path.join(inputdir, '%s-%s.hdf5' % (group, cls))
    retval = bob.io.load(filename)
    if verbose: print "[%-5s] %-6s: %8d" % (group, cls, retval.shape[0])
    return retval

  devel_real = loader('devel', 'real', datadir, verbose)
  devel_attack = loader('devel', 'attack', datadir, verbose)
  test_real = loader('test', 'real', datadir, verbose)
  test_attack = loader('test', 'attack', datadir, verbose)

  mfile = bob.io.HDF5File(machine, 'r')
  mlp = bob.machine.MLP(mfile)

  # runs the data through the MLP machine
  dev_pos = mlp(devel_real)[:,0]
  dev_neg = mlp(devel_attack)[:,0]

  # calculates the threshold
  if (mhter):
    thres = bob.measure.min_hter_threshold(dev_neg, dev_pos)
  else:
    thres = bob.measure.eer_threshold(dev_neg, dev_pos)

  # calculates the HTER on the test set using the previously calculated thres.
  tst_pos = mlp(test_real)[:,0]
  tst_neg = mlp(test_attack)[:,0]
  dev_far, dev_frr = bob.measure.farfrr(dev_neg, dev_pos, thres)
  far, frr = bob.measure.farfrr(tst_neg, tst_pos, thres)

  return ((dev_far + dev_frr) / 2., (far + frr) / 2.)

def parse_error_table(table, mhter):
  """Parses result tables and extracts the HTER for both development and test 
  sets, returning them as a tuple.
  
  The value of "mhter" is a boolean indicating if we should take the EER 
  threshold or the Min.HTER threshold performance values.
  """
  perf_line = re.compile(r'^.*HTER\s*(?P<val>\d+\.\d+)%\s*$')

  values = []
  for line in open(table, 'rt'):
    m = perf_line.match(line)
    if m: values.append(float(m.groupdict()['val']))

  if mhter: return values[3:]
  return values[:2]
