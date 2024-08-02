import pandas as pd
import matplotlib.pyplot as plt
from reweighting_tools import BDTReweighter
import numpy as np

# in this example we add negative weights for both target and original

# Generate some data distributed according to Gaussians
# We will have 2 sets of data with label 0 and 1 and each will have different values for the Gaussian parameters
  
# Parameters for Gaussian distributions

mean1 = 90
std_dev1 = 20
size1 = 100000

mean2 = 100
std_dev2 = 15
size2 = 120000

bkg_frac_1 = 0.3 # fraction of background events compared to non-bkg
over_sample_bkg_1=3 # oversample the bkg distribution and weight down the events in the subtraction

# Generate random data for each DataFrame
data1_1 = np.random.normal(mean1, std_dev1, size1)
data1_2 = np.random.normal(100, 30, int(size1*bkg_frac_1))
bkg1 = np.random.normal(100, 30, int(size1*bkg_frac_1)*over_sample_bkg_1)

data1 = np.concatenate([data1_1, data1_2])
np.random.shuffle(data1)

bkg_frac_2 = 0.3 # fraction of background events compared to non-bkg
over_sample_bkg_2=3 # oversample the bkg distribution and weight down the events in the subtraction

data2_1 = np.random.normal(mean2, std_dev2, size2)
# add some background events that we will want to subtract later on
data2_2 = np.random.normal(70, 30, int(size2*bkg_frac_2))
bkg2 = np.random.normal(70, 30, int(size2*bkg_frac_2)*over_sample_bkg_2)

data2 = np.concatenate([data2_1, data2_2])

np.random.shuffle(data2)

data2_weights = np.ones(len(data2))
data1_weights= np.ones(len(data1))
bkg1_weights = -np.ones(len(bkg1))/float(over_sample_bkg_1)
bkg2_weights = -np.ones(len(bkg2))/float(over_sample_bkg_2)

def CombineAndShuffle(samples, weights):
    X_combined = np.concatenate(samples, axis=0)
    w_combined = np.concatenate(weights, axis=0)
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    w_combined = w_combined[indices]

    return X_combined, w_combined


reweighter = BDTReweighter(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})

target, target_weights = CombineAndShuffle((data2, bkg2), (data2_weights, bkg2_weights))
original, original_weights = CombineAndShuffle((data1, bkg1), (data1_weights, bkg1_weights))

reweighter.fit(original, target, original_weights, target_weights)

original_re_weights = reweighter.predict_weights(original)
original_re_weights*=original_weights

## renormalise after fit
#normalization = w_target_weights.sum()/re_weight_1.sum()
#original_re_weights*=normalization

lim_min = min(mean1-std_dev1*4, mean2-std_dev1*4)
lim_max = max(mean1+std_dev1*4, mean2+std_dev2*4)

def MakePlot(samples=[], weights=[], labels=[], outputname='output_plot.pdf', density=False):
    plt.figure(figsize=(10, 6))
    plt.hist(samples[0], bins=40, alpha=0.5, color='r', histtype='step', label=labels[0], range=(lim_min, lim_max),weights=weights[0], density=density)
    plt.hist(samples[1], bins=40, alpha=0.5, color='b', histtype='step', label=labels[1],range=(lim_min, lim_max),weights=weights[1], density=density)
    plt.hist(samples[2], bins=40, alpha=0.5, color='g', histtype='step', label=labels[2],range=(lim_min, lim_max), weights=weights[2], density=density)
    plt.legend()
    plt.savefig(outputname)
    print('Saving plot: %s' % outputname)
    plt.close()

MakePlot([original,target,original], [original_weights, target_weights, original_re_weights], ['Original','Target','Original reweighted'], 'BDT_example.pdf')


