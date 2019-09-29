
import common

import math
import numpy as np

import scipy.stats as stats
from scipy.optimize import minimize

import pylab as plt
import matplotlib as mpl

from StochChemistry import StochChemistry
from Filament import Filament
from Motor import Motor
from Langevin import Langevin

#Global plotting params
#plt.rcParams['font.sans-serif']=["Arial"] 
#plt.rcParams['xtick.labelsize'] = 8 
#plt.rcParams['ytick.labelsize'] = 8 
#plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
#plt.rcParams.update({'figure.autolayout': True})

############################################
#
#	   STOCHFILAMENTS TESTING FUNCTIONS
#	
############################################

#	Testing simple overdamped langevin motion of individual filaments for various friction coefficients.
def langevinTestSimple(time, numruns, timestep, saveDir = "/Users/jameskomianos/Desktop/"):

	print "LANGEVIN TEST SIMPLE"
	print 

	langevin = Langevin()

	ogfric = common.fric_coeff
	fricfactors = [1E2,1E1,1E0,1E-1,1E-2]

	mean_x = []

	for fricfactor in fricfactors:

		cur_x = []
		common.fric_coeff = ogfric * fricfactor

		#create filament, mp = 0 nm, L = 2000 nm
		f = Filament(0, 2000, K=0.0)

		failflag = False
		for run in range(numruns):

			#reset to original pos
			f.resetPosition()
			if(not (langevin.overdampedLangevinSingle(f, timestep, int(time / timestep)))):

				print "Failed for friction coeff =", common.fric_coeff, " pN s / nm and timestep = ", timestep, " s."
				failflag = True
				mean_x.append((0,0))
				break

			cur_x.append(f.mp)

		if(failflag): continue

		#plot histogram of cur x
		plt.figure(6, figsize = (6,6))
		bins = max(int(max(cur_x) - min(cur_x)),1)

		plt.hist(cur_x, normed=1, bins = bins, histtype='step', fill=True, \
				 linewidth=0.0, alpha=0.5, label = r'$\mathsf{\eta = %.2e \/ pN\/ s / nm}$' % (common.fric_coeff))

		plt.xlabel(r'$\mathsf{x\/(nm)}$', fontsize=12)
		plt.ylabel(r'$\mathsf{P(x)}$', fontsize=12)

		plt.legend(prop={'size':8}, loc=1)

		#print mean, stdev, plot normal fit
		x_fit = linspace(int(min(cur_x)), int(max(cur_x)), bins)
		pdf_fitted = stats.norm.pdf(x_fit,loc=0,scale=sqrt(2 * common.kT * time / common.fric_coeff))

		plt.plot(x_fit,pdf_fitted,'r--')

		plt.savefig(saveDir + '/langtestsimple.pdf', format='pdf')

		#append mean and std error
		mean_x.append((np.mean(cur_x), np.std(cur_x) / math.sqrt(len(cur_x))))

	#reset viscosity
	common.fric_coeff = ogfric

	#check if means agree
	finalvals = []
	fricit = 0
	print "Statistics..."
	for mean, stderr in mean_x:

		fricfactor = fricfactors[fricit]
		fricit +=1

		var = (stderr * math.sqrt(len(cur_x)))**2
		print "Distribution with fric coeff = ", fricfactor*ogfric, ", Mean = ", mean, " nm, Var = ", var, " nm^2"

		finalvals.append((fricfactor*ogfric, mean, var))

	return finalvals

#Testing simple overdamped langevin motion of individual tethered filaments
def langevinTestTethered(time, timestep, saveDir = "/Users/jameskomianos/Desktop/"):

	print "LANGEVIN TEST TETHERED"
	print 

	langevin = Langevin()

	K = [1.0]#[0.001,0.01,0.1,1.0]

	mean_x = []

	colors = ['y','k','darkgreen','darkblue']

	for k in K:
		cur_x = []

		#create filament, mp = 0 nm, L = 2000 nm
		f = Filament(0, 2000, K=k)

		failflag = False

		for i in xrange(int(time/timestep)):

			if(not (langevin.overdampedLangevinSingle(f, timestep, 1))): 

				print "Failed for K =", k, " pN / nm and timestep = ", timestep, " s."
				print

				failflag = True
				mean_x.append((0,0))
				break

			cur_x.append(f.mp)

		if(failflag): continue

		#plot histogram of cur x
		plt.figure(6, figsize = (6,2))
		xbins = max(int((max(cur_x) - min(cur_x))) / 2,1)

		n,bins,patches = plt.hist(cur_x, normed=1, bins = xbins, histtype='stepfilled', \
				 linewidth=0.0, alpha=0.5, color='darkblue', label = r'$\mathsf{K_t = %g \/ pN / nm}$' % (k))

		plt.setp(patches, 'facecolor', 'darkblue', 'alpha', 0.75)

		plt.xlabel(r'$\mathsf{x\/(nm)}$', fontsize=12)
		plt.ylabel(r'$\mathsf{P(x)}$', fontsize=12)

		plt.legend(prop={'size':8}, loc=1)

		#print mean, stdev, plot normal fit

		x_fit = linspace(int(min(cur_x)), int(max(cur_x)), xbins)
		pdf_fitted = stats.norm.pdf(x_fit,loc=0,scale=sqrt(common.kT / k))

		plt.plot(x_fit,pdf_fitted,'r--')

		plt.xlim(-200,200)

		plt.savefig(saveDir + '/langtesttethered.pdf', format='pdf')

		#append mean and std error
		mean_x.append((np.mean(cur_x), np.std(cur_x) / math.sqrt(len(cur_x))))

	#check if means agree
	finalvals = []
	kit = 0
	print "Statistics..."
	for mean, stderr in mean_x:

		var = (stderr * math.sqrt(len(cur_x)))**2
		print "Distribution with K = ", K[kit]," , timestep = ", timestep, ", Mean = ", mean, " nm, Var = ", var, " nm^2"
		finalvals.append((K[kit], mean, var))

		print "Average energy = ",  0.5 * var * K[kit], " pN nm"
		kit+=1

	print

	return finalvals 

#Testing simple overdamped langevin motion of two tethered filaments
def langevinTestDoubleTethered(time, timestep, saveDir = "/Users/jameskomianos/Desktop/"):

	print "LANGEVIN TEST TWO TETHERED"
	print 

	langevin = Langevin()

	K = [1.0]#[0.001,0.01,0.1,1.0]
	colors = ['y','k','darkgreen','darkblue']

	mean_l = []

	for k in K:
		cur_l = []

		#create filaments, mp1 = 0 nm, mp2 = 4000 nm, L = 2000 nm
		f1 = Filament(0, 2000, K=k)
		f2 = Filament(4000, 2000, K=k)

		failflag = False

		for i in xrange(int(time/timestep)):

			pass1 = langevin.overdampedLangevinSingle(f1, timestep, 1)
			pass2 = langevin.overdampedLangevinSingle(f2, timestep, 1)

			if(not pass1 or not pass2): 

				print "Failed for K =", k, " pN / nm and timestep = ", timestep, " s."
				print

				failflag = True
				mean_l.append((0,0))
				break

			#calculate overlap
			cur_l.append((f1.mp + f1.L) - (f2.mp - f2.L))

		if(failflag): continue

		#plot histogram of cur x
		plt.figure(6, figsize = (6,2))
		xbins = max(int((max(cur_l) - min(cur_l))) / 2,1)

		n,bins,patches = plt.hist(cur_l, normed=1, bins = xbins, histtype='stepfilled', \
				 linewidth=0.0, alpha=0.5, color = 'darkblue', label = r'$\mathsf{K_t = %g \/ pN / nm}$' % (k))

		plt.setp(patches, 'facecolor', 'darkblue', 'alpha', 0.75)

		plt.xlabel(r'$\mathsf{l_{o}\/(nm)}$', fontsize=12)
		plt.ylabel(r'$\mathsf{P(l_{o})}$', fontsize=12)

		plt.legend(prop={'size':8}, loc=1)

		#print mean, stdev, plot normal fit
		l_fit = linspace(int(min(cur_l)), int(max(cur_l)), xbins)
		pdf_fitted = stats.norm.pdf(l_fit,loc=0,scale=sqrt(2 * common.kT / k))

		plt.plot(l_fit,pdf_fitted,'r--')

		plt.xlim(-200,200)

		plt.savefig(saveDir + '/langtesttwotethered.pdf', format='pdf')

		#append mean and std error
		mean_l.append((np.mean(cur_l), np.std(cur_l) / math.sqrt(len(cur_l))))

	#check if means agree
	finalvals = []
	kit = 0
	print "Statistics..."
	for mean, stderr in mean_l:

		var = (stderr * math.sqrt(len(cur_l)))**2
		print "Distribution with K = ", K[kit]," , timestep = ", timestep, ", Mean = ", mean, " nm, Var = ", var, " nm^2"
		finalvals.append((K[kit], mean, var))

		print "Average energy = ",  0.5 * var * K[kit], " pN nm"
		kit+=1

	return finalvals


#Testing unbound->bound steady state stochastic kinetics
def gillespieTestSimple(saveDir = '/Users/jameskomianos/Desktop/'):

	plt.figure(9, figsize = (4,4))
	plt.xlabel(r'$\mathsf{n}$', fontsize=12)
	plt.ylabel(r'$\mathsf{P(n)}$', fontsize=12)

	print "GILLESPIE TEST SIMPLE"
	print 

	k_on = 1.0
	k_off = 1.0

	cl = StochChemistry(k_on, k_off,100,0.0,0.0,0.0)

	#compute gillespie steps
	print "Test part 1..."
	i = 0
	n = []
	while (i < 1E6):
		cl.gillespieStep()

		n.append(cl.n)
		i+=1

	bins = max(int((max(n) - min(n))),1)
	plt.hist(n, normed=1, bins = bins, histtype='step', fill=True, \
			 linewidth=0.0, alpha=0.5, label = r'$\mathsf{k_{on}^{cl} = %.1f \/ s^{-1},\/ k_{off}^{cl} = %.1f \/ s^{-1} }$' % (k_on, k_off))


	#Test for mean ~10
	if(abs(50 - sum(n) / float(len(n))) > 0.1) :

		print "FAILED: Gillespie did not reach correct average value ~50."
		print "Avg n = ", sum(n) / float(len(n)) 
	else :
		print "PASSED: Correct mean reached!"
		print "Avg n = ", sum(n) / float(len(n)) 

	#Test for poisson distribution
	result = minimize(common.negLogLikelihood, x0 = np.ones(1), args=(n,), method='Powell')

	if(abs(result.x - sum(n) / float(len(n))) > 0.01) :
		print "FAILED: Correct distribution not achieved!"
		print result
	else:
		print "PASSED: Correct distribution achieved!"
		print result

	#different params
	k_on = 5.0
	k_off = 10.0

	cl = StochChemistry(k_on, k_off, 100, 0.0,0.0,0.0)

	#compute gillespie steps
	print "Test part 2..."
	i = 0
	n = []
	while (i < 1E6):
		cl.gillespieStep()
		n.append(cl.n)
		i+=1

	bins = max(int((max(n) - min(n))),1)
	plt.hist(n, normed=1, bins = bins, histtype='step', fill=True, \
			 linewidth=0.0, alpha=0.5, label = r'$\mathsf{k_{on}^{cl} = %.1f \/ s^{-1},\/ k_{off}^{cl} = %.1f \/ s^{-1} }$' % (k_on, k_off))

	#Test for mean ~33
	if(abs(33 - sum(n) / float(len(n))) > 0.1) :

		print "FAILED: Gillespie did not reach correct average value ~50."
		print "Avg n = ", sum(n) / float(len(n)) 
	else :
		print "PASSED: Correct mean reached!"
		print "Avg n = ", sum(n) / float(len(n)) 

	#Test for poisson distribution
	result = minimize(common.negLogLikelihood, x0 = np.ones(1), args=(n,), method='Powell')

	if(abs(result.x - sum(n) / float(len(n))) > 0.01) :
		print "FAILED: Correct distribution not achieved!"
		print result
	else:
		print "PASSED: Correct distribution achieved!"
		print result


	#different params
	k_on = 1.0
	k_off = 10.0

	cl = StochChemistry(k_on, k_off, 100, 0.0,0.0,0.0)

	#compute gillespie steps
	print "Test part 3..."
	i = 0
	n = []
	while (i < 1E6):
		cl.gillespieStep()
		n.append(cl.n)
		i+=1

	bins = max(int((max(n) - min(n))),1)
	plt.hist(n, normed=1, bins = bins, histtype='step', fill=True, \
			 linewidth=0.0, alpha=0.5, label = r'$\mathsf{k_{on}^{cl} = %.1f \/ s^{-1},\/ k_{off}^{cl} = %.1f \/ s^{-1} }$' % (k_on, k_off))

	#Test for poisson distribution
	result = minimize(common.negLogLikelihood, x0 = np.ones(1), args=(n,), method='Powell')

	if(abs(result.x - 9.5) > 0.1) :
		print "FAILED: Correct distribution not achieved!"
		print result
	else:
		print "PASSED: Correct distribution achieved!"
		print result

	plt.legend(prop={'size':8}, loc=1)

	plt.savefig(saveDir + '/gillespietest.pdf', format='pdf')



