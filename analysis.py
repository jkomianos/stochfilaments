
import common
import time
import ast

import math
import numpy as np
import scipy
from scipy import stats

import pylab as plt
import matplotlib as mpl
#from cycler import cycler

import cPickle as pickle
from scipy.optimize import curve_fit
from scipy.stats.kde import gaussian_kde
from scipy.interpolate import spline

from StochFilaments import StochFilaments

#Global plotting params
plt.rcParams['font.serif']=["Times New Roman"] 
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 8 
plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
plt.rcParams.update({'figure.autolayout': True})

############################################
#
#	   STOCHFILAMENTS ANALYSIS FUNCTIONS
#	
############################################

###############    SIMULATION SETUP   ####################

v_b = 3.5E-6   #binding site volume (um^3)
v   = 1.6E-3   #effective volume of diffusing cross-linkers at 1uM (um^3)

ref_e = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] #in kT 
ref_k_minus = 10.0 # in 1/s

##########################################################

#Generate l and variations of l (energy, force) vs the kinetic on/off rates
#
#	L_f = linker kinetic factor which defines the kinetic rates: K_d = L_f * k_on,0 / k_off,0
#		  where k_off,0 is 10.0 / s
#	K_t = filament tether spring constant in pN / nm
#	V_m = Motor velocity in nm / s
#	K_m = Motor spring constant in pN / nm
#	kon_m, koff_m = Motor on and off rates (1/s)
#
#
def genData(L_f, K_t, V_m, K_m, kon_m = 2.0, koff_m = 0.97, d=10, mpR=4000.0, mpL=0.0, bdMotor=False, outputDir = "./data/", numruns=250, runtime=200):

	print "L_f = ", L_f
	print "K_t = ", K_t
	print "V_m = ", V_m
	print "K_m = ", K_m

	k_minus = ref_k_minus * L_f

	#on rate based on energies
	k_plus = [(k_minus * v_b / v) * math.exp(eng) for eng in ref_e]

	#data for l
	dataL = []
	dataN = []
	dataLSquare = []

	print "Sims with L_f = ", L_f, ", K_t = ", K_t, ", V_m = ", V_m, ", K_m = ", K_m

	#Choose timestep
	timestep = 0.1E-3

	start = time.time()

	i_e = 0
	for kp in k_plus:

		sim = StochFilaments(kp, k_minus, kon_m, koff_m, K_t=K_t, K_m=K_m, V_m=V_m, d=d, mpR=mpR, mpL=mpL)

		if(bdMotor):
			sim.motor.setAsBidirectional()

		l_data, n_data, l_intervals, motor_unbindingtimes, \
		motor_walklengths, linker_unbindingtimes, linker_forcepercl, \
		r_amp, r_freq, s_amp, s_freq = sim.bigsim(numruns=numruns, runtime=runtime, plotTrajectories = True, plotVisualization=True, langevinTimeStep=timestep)

		#Save data as indexed by kp, km
		pickle.dump( l_data, open( outputDir + "/histl-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )
		pickle.dump( n_data, open( outputDir + "/histn-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )
		pickle.dump( l_intervals, open( outputDir + "/intervalsl-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )

		pickle.dump( motor_unbindingtimes, open( outputDir + "/munbindtime-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )
		pickle.dump( motor_walklengths, open( outputDir + "/mwalklength-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )

		pickle.dump( linker_forcepercl, open( outputDir + "/lforcepercl-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )
		pickle.dump( linker_unbindingtimes, open( outputDir + "/lunbindtime-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )

		pickle.dump(r_amp, open( outputDir + "/ramp-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )
		pickle.dump(r_freq, open( outputDir + "/rfreq-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )

		pickle.dump(s_amp, open( outputDir + "/samp-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )
		pickle.dump(s_freq, open( outputDir + "/sfreq-e%0.0f-Kt%0.3f-koff%0.4f.p" % (abs(ref_e[i_e]), K_t, k_minus), "wb" ) )

		print "Done simulation with k_on = ", kp, ", k_off = ", k_minus

		dataL.append(sim.getLMean())
		dataN.append(sim.getNMean())
		dataLSquare.append(sim.getLMeanSquare())

		i_e += 1

	end = time.time()

	print "Total time elapsed = ", end - start, " s"

	l   = []
	l_s = []
	n   = []
	err_l  = []
	err_ls = []
	err_n  = []

	for i in xrange(len(dataL)):

		l.append(dataL[i][0])
		err_l.append(dataL[i][1])

	for i in xrange(len(dataN)):

		n.append(dataN[i][0])
		err_n.append(dataN[i][1])

	for i in xrange(len(dataLSquare)):

		l_s.append(dataLSquare[i][0])
		err_ls.append(dataLSquare[i][1])

	#Save data as indexed by K, factor
	pickle.dump( l, open( outputDir + "/meanl-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "wb" ) )
	pickle.dump( err_l, open( outputDir + "/errmeanl-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "wb" ) )

	pickle.dump( l_s, open( outputDir + "/meanls-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "wb" ) )
	pickle.dump( err_ls, open( outputDir + "/errmeanls-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "wb" ) )

	pickle.dump( n, open( outputDir + "/meann-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "wb" ) )
	pickle.dump( err_n, open( outputDir + "/errmeann-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "wb" ) )


def plotTrajectories(energy, K_t, koff, outputDir = "/Users/jameskomianos/Code/stochfilaments/data-PROD3/data-NM/output/", saveDir = "/Users/jameskomianos/Desktop/"):

	#predicted steady state
	lo_ss = energy * common.kT * (1.0 / (1.0 + (v / v_b) * math.exp(-energy))) / (2.0 * 10.0 * K_t)
	print "Predicted steady-state overlap = ", lo_ss

	#mpl.rcParams['axes.color_cycle'] = ['y','k','darkgreen']
	#mpl.rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c']
	mpl.rcParams['axes.color_cycle'] = ['darkorange', 'darkorange', 'blueviolet', 'blueviolet', 'brown', 'brown', 'darkblue']

	pvals = []
	perrs = []

	k_minus = koff

	runtime = 200.0
	numIntervals = 200.0

	l_intervals = pickle.load( open( outputDir + "/intervalsl-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, k_minus), "rb" ) )

	if (len(l_intervals) != 0):

		intervalTime = runtime / numIntervals 

		t = []
		l = []
		devs = []
		errplus = []
		errminus = []

		intervalTime = 0
		for i in xrange(0, len(l_intervals)):

			intervalTime += runtime/ numIntervals

			lset = l_intervals[i]

			lval = np.mean(lset)
			l.append(lval)
			t.append(intervalTime)

			dev = np.std(lset) / math.sqrt(250.0)
			devs.append(dev)

			errplus.append(lval + dev)
			errminus.append(lval - dev)

			logl = [math.log(x) for x in l if x > 0]
			logt = [math.log(x) for x in t if x > 0]
			logdev = []

			index = 0
			for lval in l:
				logdev.append(0.434 * devs[index] / lval)
				index += 1

	#truncate for relevance
	#semi-log is needed for t~tlab
	index = 0
	logl_semil = []
	t_semil = []

	logl_ll = []
	logt_ll = []

	flag = "LIN_COLLECT"
	initSlope = 0.0
	tautrans = 0.0

	for tval in t:

		if(flag == "EXP_COLLECT" and tval > tautrans * 2):

			logl_semil.append(math.log(l[index]))
			t_semil.append(t[index])		

		if(flag == "LIN_COLLECT"):
			logl_ll.append(logl[index])
			logt_ll.append(logt[index])

			if(len(logl_ll) > 2):
				slope, intercept, r_value, p_value, std_err = stats.linregress(logt_ll,logl_ll)
				if(len(logl_ll) == 3):
					initSlope = slope
				elif(abs(slope - initSlope) / initSlope >= 0.10):
					tautrans = tval
					flag = "EXP_COLLECT"


		index += 1

	lo_last = l[index-1]

	print "Transition time = ", tautrans, " s" 

	#plot trajectory
	plt.figure(8, figsize = (3.5,2.5))

	#smooth out
	that = np.linspace(min(t),max(t),10000)
	lhat = spline(t,l,that)

	p = plt.plot(that, lhat, markersize=2,linewidth=2.0)
	color = p[0].get_color()

	plt.fill_between(t, errplus, errminus, color=color, alpha=0.3)

	plt.xlabel(r'$\mathsf{Time\/ (s)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\langle l_o \rangle\/(nm)}$', fontsize=12)
	plt.savefig(saveDir + '/traj.pdf', format='pdf')
	plt.xlim((0,100))

	#plot log(l) vs log(t)
	plt.figure(10, figsize = (3.5,3))
	plt.errorbar(logt, logl, fmt='o', yerr=logdev)#, mfc='white', zorder=1)
	plt.xlabel(r'$\mathsf{log(t)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{log(\langle l_o \rangle)}$', fontsize=12)
	plt.savefig(saveDir + '/trajlog.pdf', format='pdf')

	#regression
	print
	slope, intercept, r_value, p_value, std_err = stats.linregress(logt_ll,logl_ll)
	print "Log-log plot..."
	print "Observed alpha_l = ", math.exp(intercept)
	print "r^2 = ", r_value*r_value
	print "std_err = ", std_err 
	print "v = ", slope 
	
	#plot log(l) vs t
	plt.figure(9, figsize = (3,3))

	plt.errorbar(t, logl, fmt='o-', yerr=logdev)
	plt.xlabel(r'$\mathsf{Time\/ (s)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{log(\langle l_o \rangle)}$', fontsize=12)
	plt.savefig(saveDir + '/trajsemilog.pdf', format='pdf')

	slope, intercept, r_value, p_value, std_err = stats.linregress(t_semil,logl_semil)
	print
	print "Semi-log plot..."
	print "Observed alpha = ", math.exp(intercept)
	print "r^2 = ", r_value*r_value
	print "std_err = ", std_err 
	print "u = ", slope 


#plot power law data obtained
def plotPowerLawData(saveDir = "/Users/jameskomianos/Desktop/"):

	#SHORT TIME LIMIT
	energies = [6,7,8,9,10,11,12]
	v_o_cl = [2.818,8.303,8.245,10.549,8.401,9.185,11.901]
	v_o_m = [3.134,3.756,4.314,2.911,4.179,2.911,3.490]

	log_vo_cl = [math.log(x) for x in v_o_cl]
	log_vo_m = [math.log(x) for x in v_o_m]

	log_energies = [math.log(common.kT * x * (1 / (1 + (v/v_b)*math.exp(-x)))) for x in energies]

	#plot ln(eP) vs v
	plt.figure(9, figsize = (2,2))
	pyplot.locator_params(nbins=3)

	plt.errorbar(log_energies, log_vo_cl, fmt='o', color='k')
	plt.errorbar(log_energies, log_vo_m, fmt='o', color='r')

	plt.xlabel(r'$\mathsf{ln(\epsilon P_o^{cl})}$', fontsize=12)
	plt.ylabel(r'$\mathsf{ln(v_o)}$', fontsize=12)

	slope, intercept, r_value, p_value, std_err = stats.linregress(log_energies, log_vo_cl)
	print "Slope cl= ", slope
	print "R^2 cl= ", r_value*r_value
	print "err cl= ", std_err

	#show best fit line
	x = np.linspace(2.3,4,20)
	bf = [slope * val + intercept for val in x]
	plt.plot(x, bf, color='k')

	slope, intercept, r_value, p_value, std_err = stats.linregress(log_energies, log_vo_m)
	print "Slope m= ", slope
	print "R^2 m= ", r_value*r_value
	print "err m= ", std_err

	#show best fit line
	x = np.linspace(2.3,4,20)
	bf = [slope * val + intercept for val in x]
	plt.plot(x, bf,color='r')

	plt.savefig(saveDir + '/powerlawst.pdf', format='pdf')

	#LONG TIME LIMIT
	energies = [6,7,8,9,10,11,12]
	inv_tau_cl = [0.001,0.00201,0.000606,0.000267,0.000129,0.000111,3.301E-5]
	inv_tau_m = [0.00406,0.00274,0.0013,0.00100,0.000893,0.000871,0.000970]

	l_o_lab_cl = [0.29438948648108043, 0.66043406077311, 0.5704751653246907, 0.4832527226289342, 0.4112665812914953, 0.38451964896598656, 0.3647876283138238]
	l_o_lab_m = [1.8356654362530835, 1.1425291060111316, 0.7713210276697049, 0.5818464552731532, 0.4596282639642431, 0.38121786267588437, 0.34901118004983306]

	inv_tau_pred_cl = [(0.01/common.fric_coeff)*(1 + (v/v_b)*math.exp(x))**(-l_o_lab_cl[energies.index(x)] / 10.0) for x in energies]
	inv_tau_pred_m = [(0.01/common.fric_coeff)*(1 + (v/v_b)*math.exp(x))**(-l_o_lab_m[energies.index(x)] / 10.0) for x in energies]

	#logs of all
	log_inv_tau_cl = [math.log(1/x) for x in inv_tau_cl]
	log_inv_tau_m = [math.log(1/x) for x in inv_tau_m]

	log_inv_tau_pred_cl = [math.log(x) for x in inv_tau_pred_cl]
	log_inv_tau_pred_m = [math.log(x) for x in inv_tau_pred_m]

	print log_inv_tau_pred_cl, log_inv_tau_cl

	#plot ln(bc^lo/d) vs ln(inv_tau)
	plt.figure(10, figsize = (2,2))
	pyplot.locator_params(nbins=3)

	plt.errorbar(log_inv_tau_pred_cl, log_inv_tau_cl, fmt='o', color='k')
	plt.errorbar(log_inv_tau_pred_m, log_inv_tau_m, fmt='o', color='r')

	plt.xlabel(r'$\mathsf{ln(bc^{-l_o(\tau_{lab})/\Delta})}$', fontsize=12)
	plt.ylabel(r'$\mathsf{ln(1/\tau_{ss})}$', fontsize=12)

	slope, intercept, r_value, p_value, std_err = stats.linregress(log_inv_tau_pred_cl, log_inv_tau_cl)
	print "Slope cl= ", slope
	print "R^2 cl= ", r_value*r_value
	print "err cl= ", std_err

	#show best fit line
	x = np.linspace(0,2,20)
	bf = [slope * val + intercept for val in x]
	plt.plot(x, bf, color='k')

	slope, intercept, r_value, p_value, std_err = stats.linregress(log_inv_tau_pred_m, log_inv_tau_m)
	print "Slope m= ", slope
	print "R^2 m= ", r_value*r_value
	print "err m= ", std_err

	#show best fit line
	x = np.linspace(0,2,20)
	bf = [slope * val + intercept for val in x]
	plt.plot(x, bf,color='r')

	plt.savefig(saveDir + '/powerlawst.pdf', format='pdf')


#plot l and n histograms
def plotHistograms(energies, K_t, koff, outputDir = "/Users/jameskomianos/Code/stochfilaments/data-PROD2/data-NM/output/", saveDir = "/Users/jameskomianos/Desktop/"):

	# l distribution
	lstats = []
	nstats = []

	for energy in energies:

		l_data = pickle.load( open( outputDir + "/histl-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "rb" ) )
		n_data = pickle.load( open( outputDir + "/histn-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "rb" ) )

		l = [l[1] for l in l_data]
		lstats.append(l)

		n = [n[1] + 1 for n in n_data]
		nstats.append(n)


	plt.figure(3, figsize = (2,2))

	#plot l statistics
	vparts = plt.violinplot(lstats, energies, showmeans=False, showmedians=True, widths=2.0)
	plt.xlabel(r'$\mathsf{\epsilon\/(k_bT)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{l_o\/(nm)}$', fontsize=12)

	for pc in vparts['bodies']:
		pc.set_facecolor('blue')
		pc.set_edgecolor('red')

	plt.savefig(saveDir + '/hist_l.pdf', format='pdf')

	plt.figure(4, figsize = (2,2))

	#plot l statistics
	vparts = plt.violinplot(nstats, energies, showmeans=False, showmedians=True, widths=2.0)
	plt.xlabel(r'$\mathsf{\epsilon\/(k_bT)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{n(t=200s)}$', fontsize=12)

	for pc in vparts['bodies']:
		pc.set_facecolor('green')
		pc.set_edgecolor('red')

	plt.ylim((0,12))

	plt.savefig(saveDir + '/hist_n.pdf', format='pdf')

def plotCrossLinkers(energy, K_t, koff, outputDir = "/Users/jameskomianos/Code/stochfilaments/data-S4/data-NM/output/", saveDir = "/Users/jameskomianos/Desktop/"):

	linker_unbindingtimes = pickle.load( open( outputDir + "/lunbindtime-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "r" ) )
	linker_forcepercl = pickle.load( open( outputDir + "/lforcepercl-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "r" ) )

	#Get final distributions of n_p for cutoff
	n_data = pickle.load( open( outputDir + "/histn-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "rb" ) )
	n = [n[1] + 1 for n in n_data]
	n_max = int(np.mean(n) + np.std(n))

	#collect data
	n_p = range(0,n_max+1)

	t_u = [] 
	fcl = []

	for i in n_p:
		t_u.append(linker_unbindingtimes[i][1])

	for i in n_p:
		fcl.append(linker_forcepercl[i][1])

	kon = (koff * v_b / v) * math.exp(energy)
 
	#Plot unbinding times
	plt.figure(55, figsize = (2,2))
	pyplot.locator_params(nbins=3)
	plt.plot(n_p, t_u, 'c-', markersize=2, linewidth=2.0)#label=r'$\mathsf{k_{off}^{cl} = %g \/ s^{-1}}$' % (koff))

	t_theo = [0.0]
	t_theo.extend([((1/(n_p_val * kon)) * (math.exp(math.log((kon+koff)/koff)*n_p_val) - 1.0)) for n_p_val in xrange(1,n_max+1)])

	plt.plot(n_p, t_theo, 'k--', markersize=2, linewidth=2.0)

	plt.xlabel(r'$\mathsf{n_p}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\bar\tau_{n=0}\/(s)}$', fontsize=12)

	plt.legend(prop={'size':6}, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)

	plt.savefig(saveDir + '/lunbindingtimes.pdf', format='pdf')

	#Plot forces
	plt.figure(56, figsize = (2,2))
	pyplot.locator_params(nbins=4)
	plt.plot(n_p, fcl, '-', color='darkgreen', markersize=2, linewidth=2.0) #,label=r'$\mathsf{K_{t}= %g \/ pN/nm}$' % (K_t))

	plt.xlabel(r'$\mathsf{n_p}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\langle F / n \rangle \/(pN)}$', fontsize=12)

	plt.legend(prop={'size':6}, loc=1)

	plt.savefig(saveDir + '/lforcepercl.pdf', format='pdf')

def plotMotors(energies, K_t, koff, outputDir = "/Users/jameskomianos/Code/stochfilaments/data-S4/data-M/output/", saveDir = "/Users/jameskomianos/Desktop/"):

	mut = []
	mwl = []

	for energy in energies:

		motor_unbindingtimes = pickle.load( open( outputDir + "/munbindtime-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "r" ) )
		motor_walklengths = pickle.load( open( outputDir + "/mwalklength-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "r" ) )

		mut.append(motor_unbindingtimes)

		mwl.append(motor_walklengths)

	plt.figure(88, figsize = (2,2))

	#plot l statistics
	vparts = plt.violinplot(mut, energies, showmeans=False, showmedians=True, widths=2.0)
	plt.xlabel(r'$\mathsf{\epsilon\/(k_bT)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\bar \tau_u \/(s)}$', fontsize=12)

	for pc in vparts['bodies']:
		pc.set_facecolor('blue')
		pc.set_edgecolor('red')

	plt.savefig(saveDir + '/motor_unbindingtimes.pdf', format='pdf')


	plt.figure(89, figsize = (2,2))

	#plot l statistics
	vparts = plt.violinplot(mwl, energies, points=15, showmeans=False, showmedians=True, widths=2.0)
	plt.xlabel(r'$\mathsf{\epsilon\/(k_bT)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{l_w\/(nm)}$', fontsize=12)

	for pc in vparts['bodies']:
		pc.set_facecolor('green')
		pc.set_edgecolor('red')

	plt.savefig(saveDir + '/motor_walklengths.pdf', format='pdf')


#plot all scatter plots
def plotScatters(L_f, K_t, lo_i=0, outputDir = "/Users/jimmy/Code/stochfilaments/output/", saveDir = "/Users/jimmy/Desktop/"):

	#ANALYTIC DATA
	#lo for cross-linkers only (Kt=0.01)
	loa_cl = [0,0.1295,0.6966,2.7600,9.2886,26.4296,59.7827,72.1158,51.0522,37.9080,29.8891,24.6716,21.0468,18.3768,16.3464,14.7404]

	#Caclulate steady-state
	lo_ss = []
	for energy in ref_e:
		lo_ss.append(energy * common.kT * (1.0 / (1.0 + (v / v_b) * math.exp(-energy))) / (2.0 * 10.0 * K_t))

	mpl.rcParams['axes.color_cycle'] = ['y','k','darkgreen']
	#mpl.rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c']
	#mpl.rcParams['axes.color_cycle'] = ['darkorange', 'blueviolet', 'brown']

	numRuns = 250.0

	#open data
	l = pickle.load( open( outputDir + "/meanl-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "rb" ) )
	err_l = pickle.load( open( outputDir + "/errmeanl-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "rb" ) )

	l_s = pickle.load( open( outputDir + "/meanls-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "rb" ) )
	err_ls = pickle.load( open( outputDir + "/errmeanls-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "rb" ) )

	n = pickle.load( open( outputDir + "/meann-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "rb" ) )
	err_n = pickle.load( open( outputDir + "/errmeann-Lf%0.3f-Kt%0.3f.p" % (L_f, K_t), "rb" ) )

	k_minus = ref_k_minus * L_f

	initVal = l[0] * K_t
	initErr = err_l[0]

	#Plot overlap
	#plt.figure(4, figsize = (2,2))
	#pyplot.locator_params(nbins=4)
	#plt.errorbar(ref_e, l, fmt='o-', yerr=err_l, label=r'$\mathsf{K_t = %g \/ pN / nm}$' % (K_t))

	#plt.xlabel(r'$\mathsf{\epsilon\/ (k_bT)}$', fontsize=12)
	#plt.ylabel(r'$\mathsf{\langle l_{overlap} \rangle\/(nm)}$', fontsize=12)

	#plt.savefig(saveDir + '/lmeanvsg0.pdf', format='pdf')

	#Plot force
	plt.figure(5, figsize = (2,2))
	pyplot.locator_params(nbins=8)
	#plt.ylim((-0.1,5.5))
	#plt.xlim((3,15))

	#smooth data
	ehat = np.linspace(min(ref_e),max(ref_e),100)
	lhat = spline(ref_e,l,ehat)

	lhat_cl = spline(ref_e,loa_cl,ehat)
	lhat_ss = spline(ref_e,lo_ss,ehat)

	print [(x - lo_i) * K_t for x in l]

	plt.plot(ehat, [(x - lo_i) * K_t for x in lhat], markersize=2, linewidth=2.0, label=r'$\mathsf{k_{off}^{cl} = %g \/ s^{-1}}$' % (k_minus))

	#plt.plot(ehat, [(x - lo_i) * K_t for x in lhat_cl], 'k-', markersize=1, linewidth=2.0)
	#plt.plot(ehat, [(x - lo_i) * K_t for x in lhat_ss], 'k--', markersize=3, linewidth=2.0)

	plt.xlabel(r'$\mathsf{\epsilon\/ (k_bT)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\langle F \rangle\/(pN)}$', fontsize=12)

	plt.savefig(saveDir + '/fmeanvsg0.pdf', format='pdf')


def plotRatchet(energy, K_t, koff, NM=False, outputDir = "/Users/jameskomianos/Code/stochfilaments/data-PROD3/data-M/output/", saveDir = "/Users/jameskomianos/Desktop/"):

	#mpl.rcParams['axes.color_cycle'] = ['y','k','darkgreen']
	#mpl.rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c']
	mpl.rcParams['axes.color_cycle'] = ['darkorange', 'blueviolet', 'brown']

	#analytical values for ratchet e=5 (Kt=0.01,koff=1.0)
	lo_analytic = [10,20,30,40,50,60,70,80,90,100]
	r_analytic_5 = [12.352,7.824,4.691,2.517,1.012,-0.009,-0.675,-1.076,-1.282,-1.350]
	s_analytic_5 = [-2.923,4.118,11.0347,17.6424,23.8894,29.7650, 35.2777, 40.4443,45.2859, 49.8249]

	r_analytic_7 = [6.0157, 7.6932, 11.2175, 14.2239, 15.8569, 16.4682, 16.5689, 16.4518, 16.2450, 15.9997]
	s_analytic_7 = [4.3059, 8.9456, 11.7706, 13.6924, 15.0861, 16.1431, 16.9722, 17.6399, 18.1891,  18.6487]

	r_analytic_9 = [5.9099, 15.3713, 17.7765, 17.0508, 16.0860, 15.1782, 14.3439, 13.5780, 12.8737, 12.2246]
	s_analytic_9 = [2.3097, 2.7137, 2.8697, 2.9526,  3.0040, 3.0390, 3.0644, 3.0837, 3.0988, 3.1109]

	#For varying kinetics
	r_analytic_7_0 = [14.3240, 15.5972, 15.3289, 14.4647, 13.2615, 11.8276, 10.2218, 8.4818, 6.6344, 4.7000]
	r_analytic_7_1 = [14.5031, 11.4808, 9.2150, 7.4131, 5.9044, 4.6115, 3.4892, 2.5039, 1.6291, 0.8438]
	r_analytic_7_2 = [6.0157,  7.6932, 11.2175, 14.2239, 15.8569, 16.4682, 16.5689, 16.4518, 16.2450, 15.9997]
	r_analytic_7_3 = [0.7750, 1.2052, 2.6438, 5.3699, 8.7512, 11.1416, 11.9522,  11.8075, 11.3046, 10.7167]

	ramp_stats = []
	ramp_err = []
	rfreq_stats = []

	samp_stats = []
	samp_err = []
	sfreq_stats = []

	lo_set = xrange(10,150,2)

	r_amp = pickle.load( open( outputDir + "/ramp-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "r" ) )
	r_freq = pickle.load( open( outputDir + "/rfreq-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "r" ) )

	s_amp = pickle.load( open( outputDir + "/samp-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "r" ) )
	s_freq = pickle.load( open( outputDir + "/sfreq-e%0.0f-Kt%0.3f-koff%0.4f.p" % (energy, K_t, koff), "r" ) )

	#extract data

	lo_vals_ra = []
	for lo in lo_set:
		if lo in r_amp and len(r_amp[lo]) > 1:
			ramp_stats.append(sum(r_amp[lo]) / len(r_amp[lo]))
			ramp_err.append(np.std(r_amp[lo]) / sqrt(len(r_amp[lo])))
			lo_vals_ra.append(lo)

	plt.figure(100, figsize = (2.0,2.0))

	#plot r statistics
	ebar = plt.errorbar(lo_vals_ra, ramp_stats, yerr=ramp_err, markevery=2, errorevery=2, fmt='o', markersize=2, linewidth=1.0, label=r'$\mathsf{\epsilon = %g \/ pN nm}$' % (energy))
	prevcolor = ebar[0].get_color()
	#plt.plot(lo_analytic, r_analytic_9, '-', color=prevcolor, linewidth=1.0)
	plt.xlabel(r'$\mathsf{l_o\/(nm)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\langle \chi \rangle \/(nm)}$', fontsize=12)

	plt.xlim((10,100))
	plt.ylim((-5,25))

	pyplot.locator_params(axis='x',nbins=5)

	plt.savefig(saveDir + '/ramp_lo.pdf', format='pdf')

	#plt.legend(prop={'size':6}, loc='upper center', bbox_to_anchor=(0.5, 1.05),\
    #      ncol=3, fancybox=True, shadow=True)


	lo_vals_sa  = []
	for lo in lo_set:
		if lo in s_amp and len(s_amp[lo]) > 1:
			samp_stats.append(sum(s_amp[lo]) / len(s_amp[lo]))
			samp_err.append(np.std(s_amp[lo])/ sqrt(len(s_amp[lo])))
			lo_vals_sa.append(lo)

	plt.figure(101, figsize = (2.0,2.0))

	#plot r statistics
	plt.errorbar(lo_vals_sa, samp_stats, yerr=samp_err, markevery=2, errorevery=2, fmt='o', markersize=2, linewidth=1.0, label=r'$\mathsf{\epsilon = %g \/ pN nm}$' % (energy))
	prevcolor = ebar[0].get_color()
	#plt.plot(lo_analytic, s_analytic_9, '-', color=prevcolor, linewidth=1.0)
	plt.xlabel(r'$\mathsf{l_o\/(nm)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\langle \xi \rangle \/(nm)}$', fontsize=12)

	plt.xlim((10,100))
	plt.ylim((-25,70))

	pyplot.locator_params(axis='x',nbins=5)

	plt.savefig(saveDir + '/samp_lo.pdf', format='pdf')

	#plt.legend(prop={'size':6}, loc='upper center', bbox_to_anchor=(0.5, 1.05),\
    #      ncol=3, fancybox=True, shadow=True)


	lo_vals_rf = []
	for lo in lo_set:
		if lo in r_freq:
			rfreq_stats.append(r_freq[lo] / 200.0)
			lo_vals_rf.append(lo)
		else:
			rfreq_stats.append(0.0)
			lo_vals_rf.append(lo)


	plt.figure(102, figsize = (2,2))

	#plot r statistics
	plt.plot(lo_vals_rf, rfreq_stats, 'o', markersize=2, markevery=2, label=r'$\mathsf{\epsilon = %g \/ pN nm}$' % (energy))
	plt.xlabel(r'$\mathsf{l_o\/(nm)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\omega_{\chi}\/(s^{-1})}$', fontsize=12)

	plt.xlim((10,100))

	pyplot.locator_params(axis='x',nbins=5)

	plt.savefig(saveDir + '/rfreq_lo.pdf', format='pdf')

	#plt.legend(prop={'size':6}, loc='upper center', bbox_to_anchor=(0.5, 1.05),\
    #      ncol=3, fancybox=True, shadow=True)

	lo_vals_sf = []
	for lo in lo_set:
		if lo in s_freq:
			sfreq_stats.append(s_freq[lo] / 200.0)
			lo_vals_sf.append(lo)
		else:
			sfreq_stats.append(0.0)
			lo_vals_sf.append(lo)


	plt.figure(103, figsize = (2,2))

	#plot s statistics
	plt.plot(lo_vals_sf, sfreq_stats, 'o', markersize=2, linewidth=2.0, label=r'$\mathsf{\epsilon = %g \/ pN nm}$' % (energy))
	plt.xlabel(r'$\mathsf{l_o\/(nm)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\omega_{\xi} \/(s^{-1})}$', fontsize=12)

	pyplot.locator_params(nbins=4)

	plt.xlim((10,100))
	plt.ylim((-20,20))


	pyplot.locator_params(axis='x',nbins=5)

	plt.savefig(saveDir + '/sfreq_lo.pdf', format='pdf')

	#plt.legend(prop={'size':6}, loc='upper center', bbox_to_anchor=(0.5, 1.05),\
    #      ncol=3, fancybox=True, shadow=True)


	#plot velocity function: v(lo) = r(lo)w_r(lo) - s(lo)w_s(lo)
	v = []
	v_err = []
	v_r = []
	v_s = []
	lo_final_vals = []

	for lo in lo_set:

		#ramp
		if(lo in lo_vals_ra):
			r = ramp_stats[lo_vals_ra.index(lo)]
			err_r = ramp_err[lo_vals_ra.index(lo)]
		else:
			r=float('nan')
		#	continue

		#samp
		if(lo in lo_vals_sa):
			s = samp_stats[lo_vals_sa.index(lo)]
			err_s = samp_err[lo_vals_sa.index(lo)]
		else:
			s=float('nan')
		#	continue

		#rfreq
		if(lo in lo_vals_rf):
			wr = rfreq_stats[lo_vals_rf.index(lo)]
		else:
			wr=float('nan')
		#	continue

		#samp
		if(lo in lo_vals_sf):
			ws = sfreq_stats[lo_vals_sf.index(lo)]
		else:
			ws=float('nan')
		#	continue

 		#Afterward, set to zeros so we plot eff vel
 		if(math.isnan(r)):
 			r=0.0
 			err_s=0.0

 		if(math.isnan(s)):
 			s=0.0
 			err_s=0.0

 		if(math.isnan(wr)):
 			wr=0.0

 		if(math.isnan(ws)):
			ws=0.0


		if(abs(r*wr - s*ws) < 1E-6):
			v.append(float('nan'))
			v_err.append(0.0)
		else:
			v.append((r*wr - s*ws))
			v_err.append(math.sqrt((ws*s)**2 + (err_r*wr)**2))

		lo_final_vals.append(lo)

	plt.figure(104, figsize = (3.5,3.0))

	#histogram final values and append
	plt.errorbar(lo_final_vals, v, yerr=v_err, fmt='o', markevery=2, errorevery=2, markersize=4, linewidth=1.0, label=r'$\mathsf{\epsilon = %g \/ pN nm}$' % (energy))
	plt.xlabel(r'$\mathsf{l_o\/(nm)}$', fontsize=12)
	plt.ylabel(r'$\mathsf{\langle v_{eff} \rangle \/(nm/s)}$', fontsize=12)

	plt.xlim((10,120))
	plt.ylim((-4,12))

	plt.savefig(saveDir + '/effvel_lo.pdf', format='pdf')


#if __name__ == "__main__":
#	import sys
#	genData(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), d=float(sys.argv[5]), mpR=float(sys.argv[6]), mpL=float(sys.argv[7]), bdMotor=ast.literal_eval(sys.argv[8]), outputDir=sys.argv[9])

