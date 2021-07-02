"""
L.Higgins Jan 2019
Script to fit C-NEXAFS data using LMFIT software.
reset variables between runs by %reset in the ipython window

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Cursor

from xraydb import XrayDB
from lmfit import Model, minimize, Parameters, report_fit
from lmfit.model import save_modelresult
from lmfit.models import StepModel, GaussianModel, ExponentialGaussianModel

#Pyrochar-Oak-450	graphite	HTC-Oak-250	PyHTC-Oak-250+450	BC-Oak-650
file = 'APS-Sept17.csv'
sample_name = 'graphite'
# sample_name = 'graphite'
save_data = False

dat = pd.read_csv(file,comment='#')
x = dat['energy']
y = dat[sample_name] * 100
params = Parameters()
params.clear()

def chantler_bkg(x,amp,shift):
    """Definition for Chantler Background taken from DOI: https://dx.doi.org/10.18434/T4HS32"""
    xdb = XrayDB()
    return amp * (xdb.mu_chantler(element='C',energy=(x-(shift-xdb.xray_edge('C','K')[0]))) - xdb.mu_chantler(element='C',energy=250))

def stohr_step(x, amp, cen, wid, d):
    """Definition of Stohr Background (Erf x exp decay) p225 eqn 7.9"""
    from scipy.special import erf as err_function
    a = (x - cen) / wid
    return amp * (1 + err_function(a)) / 2 * np.exp(- (d * (x-cen-wid)))

def asym_gaussian(x,amplitude,center,m,b):
    """Definition of Exp Gauss taken from Stohr p218 Ref 7.1 & 7.4"""
    wid = (x*m) - b
    return amplitude*np.exp(-0.5*((x-center)/(wid/(2*np.sqrt(np.log(4)))))**2)

# erf model may be chantlet or stohr step functions
erf = Model(stohr_step, prefix='erf_')
params.update(erf.make_params())

# asg = ExponentialGaussianModel(prefix='asg1_', nan_policy='propagate')
# params.update(asg.make_params())

asg1 = Model(asym_gaussian, prefix='asg1_')
params.update(asg1.make_params())

asg2 = Model(asym_gaussian, prefix='asg2_')
params.update(asg2.make_params())

gauss = [GaussianModel(prefix='g%i_' %i, nan_policy='propagate') for i in range(1,9)]
for i in range(len(gauss)):
    params.update(gauss[i].make_params())

# Exponential decay step function to simulate the edge jump at the ionisation potential
params['erf_amp'].set(0.95, vary=False)
params['erf_cen'].set(289, vary=False)
params['erf_wid'].set(2.5, vary=False)
params['erf_d'].set(0.0049, vary=False)

# Asymetric Gaussians due to leftime broadening above the ionisation potential
params['asg1_amplitude'].set(1.0, vary=True, min=0.01)
params['asg1_center'].set(296.9, vary=False)
params['asg1_m'].set(0.575, vary=False) # increases width, set val = 0.575
params['asg1_b'].set(165.75, vary=False) # decreases width, set val = 164.75

params['asg2_amplitude'].set(4.0, vary=True, min=0.01)
params['asg2_center'].set(302.5, vary=False)
params['asg2_m'].expr = 'asg1_m'
params['asg2_b'].expr = 'asg1_b'

# Quinone
params['g1_center'].set(284.9, vary=False)
params['g1_sigma'].set(0.4, max=0.75)
params['g1_amplitude'].set(0.68, min=0)

# Aromatic Base Unit
params['g2_center'].set(285.5, vary=False)
params['g2_amplitude'].set(0.4, min=0)
params['g2_sigma'].expr = "g1_sigma"

#Phenols / Ketones
params['g3_center'].set(286.7, vary=False)
params['g3_sigma'].set(0.42, max=0.6)
params['g3_amplitude'].set(0.45, min=0)
# params['g3_sigma'].expr = "g1_sigma"

# Aliphatic
params['g4_center'].set(287.7, vary=False)
params['g4_sigma'].set(0.42, max=0.6)
params['g4_amplitude'].set(0.45, min=0)

# Carboxyl
params['g5_center'].set(288.5,vary=False)
params['g5_sigma'].set(0.42, max=0.8)
params['g5_amplitude'].set(0.45, min=0)

# Carbonyl
params['g6_center'].set(290.2, vary=False)
params['g6_sigma'].set(0.42, max=0.8)
params['g6_amplitude'].set(0.45, min=0)

# Graphite sigma*
params['g7_center'].set(291.9, vary=False)
params['g7_sigma'].set(0.42, max=1.2)
params['g7_amplitude'].set(0.45, min=0)

#BKG G1
params['g8_center'].set(294.5, vary=False)
# params['g8_sigma'].set(0.8)
params['g8_sigma'].expr = 'g7_sigma'
params['g8_amplitude'].set(1.8, min=0.8)

#BKG G2
# params['g8_center'].set(306.5, vary=False)
# # params['g8_sigma'].expr = "g8_sigma"
# params['g8_sigma'].set(0.6)
# params['g8_amplitude'].set(0.2)

#Define, Initiate then fit the model, print the results of the fit
model = erf + asg1 + asg2 + gauss[0] + gauss[1] + gauss[2] + gauss[3] + gauss[4] + gauss[5] + gauss[6] + gauss[7] #+ gauss[8]
init = model.eval(params,x=x)
output = model.fit(y,params,x=x)
print(output.fit_report())

# Plot the results
fig, gridspec = output.plot(xlabel='Energy Loss (eV)', ylabel='Norm a.u.', datafmt='-')
comps = output.eval_components(x=x)
plt.plot(x,comps['erf_'],'k--',label='chantler_bkg')
plt.plot(x,comps['asg1_'],'k--',label='Asym Gaussian 1')
plt.plot(x,comps['asg2_'],'k--',label='Asym Gaussian 2')
plt.plot(x,comps['g1_'],'b--',label='g1')
plt.plot(x,comps['g2_'],'b--',label='g2')
plt.plot(x,comps['g3_'],'g--',label='g3')
plt.plot(x,comps['g4_'],'g--',label='g4')
plt.plot(x,comps['g5_'],'r--',label='g5')
plt.plot(x,comps['g6_'],'r--',label='g6')
plt.plot(x,comps['g7_'],'r--',label='g7')
plt.plot(x,comps['g8_'],'c--',label='g8')
# plt.plot(x,comps['g9_'],'c--',label='g9')
plt.legend(loc='best',bbox_to_anchor=(0.5, 1.05),ncol=4,fancybox=True)
plt.show()

if save_data:
    comps['energy'] = dat['energy'].values
    comps['signal'] = dat['PyHTC-Oak-250+450'].values*100
    save_dat = pd.DataFrame.from_dict(comps)
    save_loc = file.split('.')[0] + '_' + sample_name + '_' + 'fit_results.csv'
    save_loc1 = file.split('.')[0] + '_' + sample_name + '_' + 'fit_statistics.txt'
    save_dat.to_csv(save_loc,header=True)
    fit_stats = open(save_loc1, "w")
    fit_stats.write(output.fit_report())
    fit_stats.close()
    print('Wrote fit_results and fit_statistics to current Dir!')
