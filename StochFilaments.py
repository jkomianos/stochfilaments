
import common
import sys
from collections import defaultdict

import math
import numpy as np

import pylab as plt
import matplotlib as mpl

from Filament import Filament
from StochChemistry import StochChemistry
from StochChemistry import Reaction
from Motor import Motor

from Langevin import Langevin

#Global plotting params
#plt.rcParams['font.serif']=["Times New Roman"] 
#plt.rcParams['xtick.labelsize'] = 8
#plt.rcParams['ytick.labelsize'] = 8 
#plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
#plt.rcParams.update({'figure.autolayout': True})


#	This class represents the main stochastic simulation driver.
#	Runs stochastic simulation with given parameters, tracks all 
#	relevant parameters, outputs and plots relevant parameters.
#
class StochFilaments:

	#MEMBER VARIABLES

	#Two filaments
	r_fil = None
	l_fil = None

	#Motor
	motor = None

	#Langevin integrator
	langevin = None

	#chemical stochastic simulator
	sc = None

	#constant, user defined
	d = 0.0       #Distance between cross-links, which defines 
				  #discrete steps of possible cross-linker binding (nm)

	#simulation observables
	l = 0		  #Overlap of filaments (nm)

	tau = 0.0     #Gillespie time

	numIntervals = 50    #For now 

	#other constants

	#Flags for different function
	LANGONLY = False 	 #Langevin dynamics only
	GILLONLY = False     #Stochastic simulation only

	#Holding data
	l_data = []
	n_data = []

	l_intervals = []

	#power spike amplitudes and frequencies
	RAmp = defaultdict(list)
	RFreq = defaultdict(int)

	SAmp = defaultdict(list)
	SFreq = defaultdict(int)

	#motor data
	motor_unbindingtimes = []
	motor_walklengths = []

	#linker data
	linker_unbindingtimes = []
	linker_forcepercl = []

	#	The constructor simply initializes all input variables.
	def __init__(self, kon_l, koff_l, kon_m, koff_m, mpR=4000.0, mpL=0.0, L=2000.0, \
				 d=10.0, K_t=0.00, K_m=0.00, V_m=0.00):
		
		#initialize filaments
		self.r_fil = Filament(mpR, L, K=K_t) 
		self.l_fil = Filament(mpL, L, K=K_t) 

		self.motor = Motor(K_m)

		self.langevin = Langevin()

		self.d = d

		#create stochastic simulator
		#initially 
		self.sc = StochChemistry(kon_l, koff_l, self.computeOverlap(), \
				  kmp=kon_m, kmm = koff_m, kmw=V_m / self.motor.step_size)

		#start with n = 0
		self.n = 0

	#	Reset mipoints of filaments to original configuration,
	#	used when running multiple trajectories.
	#
	def reinitialize(self):

		self.tau = 0

		self.r_fil.resetPosition()
		self.l_fil.resetPosition()

		self.sc.reinitialize()

		self.motor.reinitialize()

		self.sc.np = self.computeOverlap()

	#	Set to only simulate gillespie chemical dynamics.
	#
	def setGillespieOnly(self):
		self.GILLONLY = True


	# 	Set to only simulate langevin thermal dynamics.
	#
	def setLangevinOnly(self):
		self.LANGONLY = True


	#	Compute overlap as in Langevin.py
	#	This function also computes and returns the number of possible cross-links (np) as:
	#	
	#	np = int(max(l,0) / d)
	#
	def computeOverlap(self):

		self.l = self.langevin.computeOverlap(self.l_fil, self.r_fil)

		return int(max(self.l,0) / self.d)


	#Compute the overall force applied to the passive cross-linkers. 
	#Note that it is assume that the system is mechanically equilibrated upon a 
	#Cross-linker binding event, so that abs(F_rs) = abs(F_ls) = F_s
	#This force contains a spring (s) as well as a motor contribution (m), as
	#
	#	F_cl = abs(F_m - F_s)
	#
	def computeLinkerForce(self):

		#Spring contribution
		F_s = abs(self.r_fil.K * (self.r_fil.mp - self.r_fil.eqmp))
		#Motor contribution
		F_m = self.motor.calculateForce(self.l)

		return abs(F_m - F_s)

	#Equilibrate the left and right filaments based on their forces experienced
	#Will calculate an average tensile force on both filaments, and update the midpoints
	#accordingly to represent a fast equilibration process
	def equilibrate(self):

		#calc average contractile force
		avg_f = (-self.r_fil.K * (self.r_fil.mp - self.r_fil.eqmp) + self.l_fil.K * (self.l_fil.mp - self.l_fil.eqmp)) / 2.0

		#update midpoints
		self.r_fil.mp = (- avg_f / self.r_fil.K) + self.r_fil.eqmp
		self.l_fil.mp = (avg_f / self.l_fil.K) + self.l_fil.eqmp

	#	Compute overdamped langevin dynamics for a number of timesteps. 
	#	Wrapper function for Langevin.overdampedLangevin()	
	#
	#	Also has an added side effect of saving the resulting overlap during all steps
	#
	def overdampedLangevin(self, timestep, numsteps, verbose=False):

		#calculate motor force
		if(verbose):
			self.computeOverlap()
			print "Overlap before = ", self.l
			print timestep, numsteps

		#right and left filament
		if(not self.langevin.overdampedLangevin(self.l_fil, self.r_fil, self.motor, timestep, numsteps)):
			return SystemError

		if(verbose):
			self.computeOverlap()
			print "Overlap after= ", self.l

	#	Compute standard langevin dynamics for a number of timesteps 
	#	Wrapper function for Langevin.standardLangevin()	
	#
	def standardLangevin(self, timestep, numsteps, verbose=False):

		if(verbose):
			self.computeOverlap()
			print "Overlap before = ", self.l

		#right and left filament
		if(not self.langevin.standardLangevin(self.l_fil, self.r_fil, self.motor, timestep, numsteps)):
			return SystemError

		if(verbose):
			self.computeOverlap()
			print "Overlap after= ", self.l

	def collectTrajectoryData(self, l, n, l_only = False, n_only = False):
	
		if(not n_only):
			l.append((self.tau, self.l))

		if(not l_only):
			if(len(n) != 0):
				n.append((self.tau - 1E-5, n[len(n)-1][1]))

			n.append((self.tau, self.sc.n))


	def collectVisualizerData(self, linker_data_v, motor_data_v):

		#Append to event visualizer
		#Channel 1 for n = 0
		if(self.sc.n == 0):
			if(len(linker_data_v) != 0 and linker_data_v[len(linker_data_v) - 1][1] == 2):
				linker_data_v.append((self.tau-1E-5,2))
			linker_data_v.append((self.tau,1))
		#Channel 2 for n != 0
		else:
			if(len(linker_data_v) != 0 and linker_data_v[len(linker_data_v) - 1][1] == 1):
				linker_data_v.append((self.tau-1E-5,1))
			linker_data_v.append((self.tau, 2))
				
		#Channel 4 for nm = 0
		if(self.sc.nm == 0):

			if(len(motor_data_v) != 0 and motor_data_v[len(motor_data_v) - 1][1] != 4):
				motor_data_v.append((self.tau-1E-5,motor_data_v[len(motor_data_v) - 1][1]))
			motor_data_v.append((self.tau, 4))

		else:
			#Channel 5 for contractile
			if(self.motor.sign == 1):
				if(len(motor_data_v) != 0 and motor_data_v[len(motor_data_v) - 1][1] != 5):
					motor_data_v.append((self.tau-1E-5,motor_data_v[len(motor_data_v) - 1][1]))

				motor_data_v.append((self.tau, 5))
			#Channel 3 for extensile
			else:
				if(len(motor_data_v) != 0 and motor_data_v[len(motor_data_v) - 1][1] != 3):
					motor_data_v.append((self.tau-1E-5,motor_data_v[len(motor_data_v) - 1][1]))

				motor_data_v.append((self.tau, 3))

	def sim(self, runtime=0.0, gillespieOnly=False, langevinOnly=False, langevinTimeStep=0.1E-3):

		#Data to save from the end of trajectories
		l = []
		n = []

		#Event visualizer data
		#linkers n=0 and n!=0 have channels 1 and 2
		#motors walking-, unbound, and walking + have channels 3,4,5
		linker_data_v = []
		motor_data_v = []

		#motor data
		motor_walklengths = []
		motor_unbindingtimes = []

		#linker unbinding times. WIll be in a dict (n_p, tau_n=0)
		#linker force per cross-link. Will be in a dict (np, F / n=0)
		linker_unbindingtimes = defaultdict(list)
		linker_forcepercl = defaultdict(list)

		#Boolean flag for finishing loop
		finish = False

		#Boolean flag for tracking linker unbinding time
		tau_cl_start = 0.0

		#Simulation cases 
		if(self.LANGONLY): #Langevin movement of filaments with fixed timestep

			self.motor.bind(self.l)

			while(self.tau < runtime):

				#Bind the motor
				self.computeOverlap()

				self.collectTrajectoryData(l, n, l_only=True)

				self.tau += langevinTimeStep
				self.overdampedLangevin(langevinTimeStep, 1)

		if(self.GILLONLY): #Gillespie stochastic chemistry only
			while(self.tau < runtime):

				self.collectTrajectoryData(l, n, n_only=True)

				dtau, rxn = self.sc.gillespieStep()
				self.tau += dtau

		if(not self.GILLONLY and not self.LANGONLY): #Typical protocol

			#slip and ratchet event tags
			slip=False
			ratchet=False

			prevSlip=False
			prevRatchet=False

			while(self.tau < runtime):

				self.collectTrajectoryData(l, n)

				#Append to event visualizer
				self.collectVisualizerData(linker_data_v, motor_data_v)

				#if we set the finish flag, do langevin steps until the end
				if(finish):
					while (self.tau < runtime):

						self.tau += langevinTimeStep
						self.overdampedLangevin(langevinTimeStep, 1)

						self.collectTrajectoryData(l,n,l_only=True)

						self.sc.np = self.computeOverlap()	

					#append the last point and return
					self.collectTrajectoryData(l, n)

					self.collectVisualizerData(linker_data_v, motor_data_v)
					return l, n, linker_data_v, motor_data_v, motor_walklengths, motor_unbindingtimes, \
						   linker_forcepercl, linker_unbindingtimes

				#Compute current force
				F_l = self.computeLinkerForce()
				self.computeOverlap()
				F_m = self.motor.calculateForce(self.l)

				#Gillespie step
				dtau, rxn = self.sc.gillespieStep(F_l=F_l, F_m=F_m, motor_sign=self.motor.sign)

				#If no reaction propensity, langevin dynamics
				if(dtau < 1E-15):

					finish = True
					continue

				else:	
					self.tau += dtau
					self.collectTrajectoryData(l, n)

					#Append to event visualizer
					self.collectVisualizerData(linker_data_v, motor_data_v)

					#Record linker data
					if(rxn == Reaction.L_BINDING and self.sc.n == 1):
						tau_start_cl = self.tau

						#if ratcheting was occuring, save data
						if(ratchet):

							self.RAmp[int(initialLo)].append(self.l - initialLo)
							self.RFreq[int(initialLo)] += 1

							ratchet=False

						if(slip):

							self.SAmp[int(initialLo)].append(-(self.l - initialLo))
							self.SFreq[int(initialLo)] += 1

							slip=False


					if(rxn == Reaction.L_UNBINDING and self.sc.n == 0):

						linker_unbindingtimes[self.sc.np].append(self.tau - tau_start_cl)

						#also record ratcheting data
						if(self.motor.bound):

							ratchet = True
							initialLo = self.l

						else:
							slip = True
							initialLo = self.l


					#If cross-link bound, record data
					if(rxn == Reaction.L_BINDING):
						#First, equlibrate filaments
						self.equilibrate()
						linker_forcepercl[self.sc.np].append(self.computeLinkerForce() / self.sc.n)

					#Update status of motor if bound / unbound
					if(rxn == Reaction.M_UNBINDING):
						self.motor.unbind(self.l)

						#save walk length and birth time
						motor_walklengths.append(self.motor.lo_eq - self.motor.lo_eq_0)
						motor_unbindingtimes.append(self.tau - self.motor.birthTime)

					if(rxn == Reaction.M_BINDING):
						self.motor.bind(self.l)
						self.motor.K = self.sc.Nb(F_m=0) * self.motor.K_s / 2.0

						self.motor.birthTime = self.tau

					if(rxn == Reaction.M_WALKING):
						self.motor.walk(self.l_fil.L)
						F_m = abs(self.motor.calculateForce(self.l))
						self.motor_K = self.sc.Nb(F_m=F_m) * self.motor.K_s / 2.0

				#If no cross-linkers, langevin dynamics until next reaction time
				while(self.sc.n == 0):

					F_l = self.computeLinkerForce()
					self.computeOverlap()
					F_m = self.motor.calculateForce(self.l)

					#compute next gillespie steps
					dtau, rxn = self.sc.gillespieStep(F_l=F_l, F_m=F_m, motor_sign=self.motor.sign)

					#If reaction is over time, run langevin dynamics to runtime (next loop)
					if(dtau < 1E-15 or self.tau > runtime):

						finish = True
						break

					#Langevin dynamics for time dtau
					else:
						nextTau = self.tau + dtau 
						while (self.tau < nextTau):

							self.tau += langevinTimeStep
							self.overdampedLangevin(langevinTimeStep, 1)

							self.collectTrajectoryData(l,n,l_only=True)

							self.sc.np = self.computeOverlap()	

						#Append to event visualizer
						self.collectVisualizerData(linker_data_v, motor_data_v)

					#Record linker data
					if(rxn == Reaction.L_BINDING and self.sc.n == 1):
						tau_start_cl = self.tau

						#if ratcheting was occuring, save data
						if(ratchet):

							self.RAmp[int(initialLo)].append(self.l - initialLo)
							self.RFreq[int(initialLo)] += 1

							prevRatchet=True
							ratchet=False

						if(slip):

							self.SAmp[int(initialLo)].append(-(self.l - initialLo))
							self.SFreq[int(initialLo)] += 1

							prevSlip=True
							slip=False


					#If cross-link bound, record data
					if(rxn == Reaction.L_BINDING):
						#First, equlibrate filaments
						self.equilibrate()
						linker_forcepercl[self.sc.np].append(self.computeLinkerForce() / self.sc.n)

					#Update status of motor if bound / unbound
					if(rxn == Reaction.M_UNBINDING):
						self.motor.unbind(self.l)

						#save walk length and birth time
						motor_walklengths.append(self.motor.lo_eq - self.motor.lo_eq_0)
						motor_unbindingtimes.append(self.tau - self.motor.birthTime)

					if(rxn == Reaction.M_BINDING):
						self.motor.bind(self.l)
						self.motor.K = self.sc.Nb(F_m=0) * self.motor.K_s / 2.0

						self.motor.birthTime = self.tau

					if(rxn == Reaction.M_WALKING):
						self.motor.walk(self.l_fil.L)
						F_m = abs(self.motor.calculateForce(self.l))
						self.motor_K = self.sc.Nb(F_m=F_m) * self.motor.K_s / 2.0

					#If theres no overlap after langevin dynamics and a 
					#linker reaction happened, reverse it and continue
					if(rxn == Reaction.L_BINDING and self.sc.np == 0):
						self.sc.n -= 1
						linker_forcepercl[self.sc.np].pop()

						if(prevRatchet):

							self.RAmp[int(initialLo)].pop()
							self.RFreq[int(initialLo)] -= 1

							prevRatchet=False

						if(prevSlip):

							self.SAmp[int(initialLo)].pop()
							self.SFreq[int(initialLo)] -= 1

							prevSlip=False

					else:
						prevRatchet=False
						prevSlip=False


		#distributions of l and n, visualizer data, motor data
		return l, n, linker_data_v, motor_data_v, motor_walklengths, motor_unbindingtimes, linker_forcepercl, linker_unbindingtimes

	def bigsim(self,numruns=0,runtime=0.0, plotTrajectories=False, plotVisualization=False, langevinTimeStep=0.1E-3):

		#reset
		self.l_data = []
		self.motor_unbindingtimes = []
		self.motor_walklengths = []

		#Stored as (num events to average, current average)
		self.linker_unbindingtimes = [(0,0) for x in xrange(100)]
		self.linker_forcepercl = [(0,0) for x in xrange(100)]

		self.RAmp = defaultdict(list)
		self.RFreq = defaultdict(int)

		self.SAmp = defaultdict(list)
		self.SFreq = defaultdict(int)

		interval = runtime / self.numIntervals
		self.l_intervals = [[] for x in xrange(self.numIntervals)]

		i = 0
		while (i < numruns):

			#run sim
			l, n, linkerV, motorV, motor_walklengths, motor_unbindingtimes, \
			linker_forcepercl, linker_unbindingtimes = self.sim(runtime=runtime, langevinTimeStep=langevinTimeStep)

			#plot trajectories
			if(plotTrajectories): self.plotTrajectory(l, n)

			#plot event visualization
			if(plotVisualization): self.plotVisualization(linkerV, motorV)

			#Collect cross-linker data
			self.getLinkerData(linker_forcepercl, linker_unbindingtimes)

			#save last point
			self.l_data.append(l[len(l) - 1])
			self.n_data.append(n[len(n) - 1])

			self.getIntervalData(l, runtime)

			self.motor_unbindingtimes.extend(motor_unbindingtimes)
			self.motor_walklengths.extend(motor_walklengths)

			#reset
			self.reinitialize()

			i+=1

		return self.l_data, self.n_data, self.l_intervals, \
			   self.motor_unbindingtimes, self.motor_walklengths, self.linker_unbindingtimes, self.linker_forcepercl, \
			   self.RAmp, self.RFreq, self.SAmp, self.SFreq

	def getIntervalData(self, l, runtime):

		interval = runtime / self.numIntervals
		ti = 0

		for time, lval in l:

			if(time > interval and ti < self.numIntervals):
				self.l_intervals[ti].append(lval) 

				ti += 1
				interval += runtime / self.numIntervals

	#Will update global averages for linker data using current trajectory data
	def getLinkerData(self, linker_forcepercl, linker_unbindingtimes):

		#forces per cl. This needs to be a moving average since there are a ton of vals
		for np, forcelist in linker_forcepercl.iteritems():

			if(np < 0 or np > 100):
				continue

			#get current avg
			N, avg = self.linker_forcepercl[np]

			#how many entries we are adding
			M = len(forcelist)

			if(N != 0):

				for val in forcelist:
					avg -= avg / N
					avg += val / N
					N += 1

				self.linker_forcepercl[np] = (N, avg)

			else:
				if(M != 0):
					self.linker_forcepercl[np] = (M, float(sum(forcelist)) / M)


		#This will also be a moving average
		for np, unbindinglist in linker_unbindingtimes.iteritems():

			if(np < 0 or np > 100):
				continue

			N, avg = self.linker_unbindingtimes[np]

			M = len(unbindinglist)

			if(N != 0):
				for val in unbindinglist:
					avg -= avg / N
					avg += val / N

					N += 1

				self.linker_unbindingtimes[np] = (N, avg)

			else:
				if(M != 0):
					self.linker_unbindingtimes[np] = (M, float(sum(unbindinglist)) / M)

	def getLMean(self):

		if (len(self.l_data) != 0):

			l = [l[1] for l in self.l_data]
			mean_l = sum(l) / float(len(l))

			#stdev
			error = np.std(l) / math.sqrt(len(l))

			return (mean_l, error)

	def getLMeanSquare(self):

		if (len(self.l_data) != 0):

			l_s = [math.pow(l[1],2) for l in self.l_data]
			mean_l_s = sum(l_s) / float(len(l_s))

			#Max likelihood estimate error
			error = np.std(l_s) / math.sqrt(len(l_s))

			return (mean_l_s, error)

	def getNMean(self):

		if (len(self.n_data) != 0):

			n = [n[1] for n in self.n_data]
			mean_n = sum(n) / float(len(n))

			#Max likelihood estimate error
			error = np.std(n) / math.sqrt(len(n))

			return (mean_n, error)


	def plotTrajectory(self, l, n):

		#take 0.1s intervals from l
		times = []
		lvals = []
		plvals = []

		timecounter = 0.0
		for time, lval in l:

			if (time > timecounter):

				lvals.append(lval)
				times.append(timecounter)

				if(timecounter > 0.0):
					plvals.append(((0.5 * (lval)**2 * self.l_fil.K) - 0.5 * lvals[len(plvals) - 1]**2 * self.l_fil.K) / 1.0)

				else:
					plvals.append(0.0)

				timecounter += 1.0

		#append last val
		times.append(200.0)
		lvals.append(lvals[len(lvals) - 1])
		plvals.append(((0.5 * lvals[len(plvals) - 1]**2 * self.l_fil.K) - 0.5 * lvals[len(plvals) - 2]**2 * self.l_fil.K) / 1.0)

		#plot trajectories of L
		if (len(l) != 0):
			plt.figure(0, figsize = (6,2))
			plt.plot(times, lvals, '-', color='y',label = r'$\mathsf{K_t = %g \/ pN / nm}$' % (self.l_fil.K))

			plt.xlabel(r'$\mathsf{Time (s)}$', fontsize=12)
			plt.ylabel(r'$\mathsf{l_o\/(nm)} $', fontsize=12)
			plt.legend(prop={'size':8}, loc=0)

			plt.xlim(0, 200.0)

			plt.savefig("/users/jameskomianos/Desktop/traj_l.pdf", format='pdf')

			#Plot force and power
			plt.figure(20, figsize = (6,2))

			plt.plot(times, [self.l_fil.K * x for x in lvals], '-',label = r'$\mathsf{k_{off}^{cl} = 0.01 (s^{-1})}$'  )

			plt.xlabel(r'$\mathsf{Time (s)}$', fontsize=12)
			plt.ylabel(r'$\mathsf{F\/(pN)} $', fontsize=12)
			#plt.legend(prop={'size':8}, loc=0)

			plt.xlim(0, 200.0)

			plt.savefig("/users/jameskomianos/Desktop/traj_f.pdf", format='pdf')

			plt.figure(21, figsize = (6,2))

			plt.plot(times, [x / common.kT for x in plvals], '-', label = 'Filament')

			plt.xlabel(r'$\mathsf{Time (s)}$', fontsize=12)
			plt.ylabel(r'$\mathsf{P\/(k_bT / s)} $', fontsize=12)
			plt.legend(prop={'size':8}, loc=0)

			plt.xlim(0, 200.0)

			plt.savefig("/users/jameskomianos/Desktop/traj_p.pdf", format='pdf')

		#take 1s intervals from l
		times = []
		nvals = []

		#plot trajectories of N
		if(len(n) != 0):
			plt.figure(1, figsize = (6,2))
			plt.plot(times, nvals, '-', label = r'$\mathsf{k_{off} = 10.0\/s^{-1}}$')

			plt.xlabel(r'$\mathsf{Time (s)}$', fontsize=12)
			plt.ylabel(r'$\mathsf{n}$', fontsize=12)
			plt.legend(prop={'size':8}, loc=0)

			plt.xlim(0, 200.0)

			plt.savefig("/users/jameskomianos/Desktop/traj_n.pdf", format='pdf')


	def plotVisualization(self, linkerV, motorV):

		plt.figure(10, figsize = (6,2))

		plt.plot([v[0] for v in linkerV], [v[1] for v in linkerV], '-', color='k')
		plt.plot([v[0] for v in motorV], [v[1] for v in motorV], '-', color='k')

		plt.yticks([0,1,2,3,4,5,6], ['',r'$\mathsf{n=0}$',r'$\mathsf{n\neq0}$',r'$\mathsf{MB-}$',r'$\mathsf{MU}$',r'$\mathsf{MB+}$',''])

		plt.xlabel(r'$\mathsf{Time (s)}$', fontsize=12)
		plt.ylabel(r'$\mathsf{State}$', fontsize=12)

		plt.xlim(0, 200.0)

		plt.savefig("/users/jameskomianos/Desktop/viz.pdf", format='pdf')



	

