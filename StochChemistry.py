
import math
import numpy as np

import common

#Enumerations for reaction events
class Reaction:

	L_BINDING, L_UNBINDING, M_BINDING, M_UNBINDING, M_WALKING, NONE = range(1,7)

# 	Controls stochastic cross-linker binding and unbinding simulation,
#	Which includes both passive and active (motor) linkers.
# 	Tracks relevant variables, performs gillespie simulations
#	for cross-linker binding and unbinding reactions.
#
class StochChemistry:

	#MEMBER VARIABLES

	k_linker_plus = 0.0  #Passive cross-linker on-rate
	k_linker_minus = 0.0 #Passive cross-linker off-rate
 
	k_motor_plus = 0.0	 #Motor on-rate
	k_motor_minus = 0.0	 #Motor off-rate

	k_motor_walk = 0.0   #Motor walking rate

	n = 0         #Number of passive cross-links bound at the given instant

	np = 0        #Number of possible passive cross-links
				  #This is found by the relation np = int(l / d)
				  #Where l is the filament overlap and d is the
				  #distance between neighboring cross-links

	nm = 0		  #Number of bound motors (0 or 1)

	#parameters for mechanochemistry
	
	N_t = 10	  #Number of available heads in motor filamnt per side of ensemble

	#Mechanosensitivity
	x_l = 0.5	  #Unbinding distance for passive cross-linker slip bond
	x_m = 2.5     #Unbinding distance for motor catch bond

	beta = 2.0	  #Head binding parameter
	rho = 0.1	  #Duty ratio of motor heads

	#constants for walking rate updating
	#We assume here that the F_s = F_s,single * N_heads * rho(F_s,single) =12 pN for each side,
	#and alpha = 0.2 as outlined in the paper. 
	F_s   = 24   
	alpha = 0.2 

	#Motor rates for single head
	kmp_s = 0.2
	kmm_s = 1.7

	# 	Constructor sets on and off rates, walking rate
	def __init__(self, klp, klm, np, kmp, kmm, kmw):

		self.k_linker_plus = klp
		self.k_linker_minus = klm
		self.np = np

		self.k_motor_plus = kmp
		self.k_motor_minus = kmm

		self.k_motor_walk = kmw

	#	Reset tau
	#
	def reinitialize(self):
		self.n = 0

		self.nm = 0

	#Calculates the number of bound heads for the motor filament as
	#
	#	N_b(F) = rho * N_t + beta * F_m / N_t
	#
	def Nb(self, F_m=0):

		return self.rho * self.N_t + self.beta * F_m / self.N_t


	#	Make a Gillespie chemical step based on all reaction rates
	# 	Follows the standard Gillespie algorithm.
	#
	#	Can input a force which will update the passive cross-link 
	#	unbinding reaction mechanochemically with the following form:
	#
	#	a = a_0 * exp(F_l x /(n kT))
	#
	#	Returns a pair of next time step and reaction event enum.
	def gillespieStep(self, F_l=0, F_m=0, motor_sign=1):

		#Propensities
		a_linker_minus = 0.0
		a_linker_plus  = 0.0
		a_motor_plus   = 0.0
		a_motor_minus  = 0.0
		a_motor_walk   = 0.0

		#Linker rates
		a_linker_plus = self.k_linker_plus * (self.np - self.n)

		if(self.n != 0): 
			#Update unbinding of cross-linkers based on classic slip form:
			#
			#	k_u = k_u,0 * exp(F_l * x_m / (n * kT))
			#
			#	where F_l is a pulling force in any direction
			#
			a_linker_minus = self.k_linker_minus * self.n * math.exp(F_l * self.x_l / (self.n * common.kT))
		else: 
			a_linker_minus = 0.0

		a_motor_plus  = self.k_motor_plus * (1 - self.nm)

		#Update unbinding of motor based on classic catch form:
		#
		#	k_u = k_u,0 * exp(-F_m * x_m / (rho * N_t * kT))
		#
		#	where F_m is a force in any direction, with
		#	a ceiling on the exponential factor to avoid blow ups.
		#
		F_u_eff = abs(F_m)

		a_motor_minus = self.k_motor_minus * self.nm * max(1.0/10.0, math.exp(- F_u_eff * self.x_m / (self.Nb(F_u_eff) * common.kT)))

		#Update velocity of motor based on classic Hill-type relation:
		#
		#	v = v_0 * (F_s - F_m) / (F_s + F_m / alpha)
		#
		#	where F_m is a positive force opposite the direction of walking.
		#
		if(motor_sign * F_m < 0.0):
			F_w_eff = abs(F_m)
		else:
			F_w_eff= 0.0

		a_motor_walk = max(0.0, 2 * self.k_motor_walk * self.nm * (self.F_s - F_w_eff) / (self.F_s + F_w_eff / self.alpha))

		#Total system propensity
		a_total = a_linker_plus + a_linker_minus + a_motor_plus + a_motor_minus + a_motor_walk

		if(abs(a_total) < 1E-10): 
			return 0.0, Reaction.NONE

		#generate tau step, increment global time
		tau_step = np.random.exponential(scale=1.0/a_total)

		#find which reaction happened
		mu = a_total*np.random.uniform()
		rates_sum = 0

		#Linkers
		rates_sum += a_linker_plus
		if(rates_sum> mu): 
			self.n+=1 
			return tau_step, Reaction.L_BINDING

		rates_sum += a_linker_minus
		if(rates_sum> mu): 
			self.n-=1 
			return tau_step, Reaction.L_UNBINDING

		#Motors
		rates_sum += a_motor_plus
		if(rates_sum> mu): 
			self.nm+=1 
			return tau_step, Reaction.M_BINDING

		rates_sum += a_motor_minus
		if(rates_sum> mu): 
			self.nm-=1 
			return tau_step, Reaction.M_UNBINDING

		rates_sum += a_motor_walk
		if(rates_sum> mu): 
			return tau_step, Reaction.M_WALKING

