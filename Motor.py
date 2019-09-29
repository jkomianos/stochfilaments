
import common

import math
import numpy as np

#	This class represents a bipolar myosin motor.
#	It is implemented as a time-varying harmonic potential.

#	This motor can be abstractly imagined as overlap-dependent potential.
#	The equilibrium overlap length of the two filaments is increased
#	as a function of time, providing force between the two filaments. The walking 
#	velocity, which determines the rate of increase of the equilibrium overlap, 
#	can be force-dependent.
#
#	An binding event of this motor simply resets the equilibrium length.
#

class Motor:

	#MEMBER VARIABLES
	K = 0.0		#Eff spring constant of motor (pN / nm)
	K_s = 0.0	#Spring constant of single motor head (pN / nm)

	lo_eq = 0.0	#Equilibrium overlap of two filaments (nm)
				#This will be time-varied

	lo_eq_0 = 0.0 #Initial eq overlap, to calculate walk length

	bound=False	#Boolean specifying bound state, which controls updating

	#Sign of directionality. If 1, contractile, if -1, extensile.
	bidirectional = False
	sign = 1

	step_size = 5.0 	#5nm steps for the ensemble
	birthTime = 0.0 	#For tracking unbinding times

	#	Initializes motor and sets all relevant parameters
	#
	def __init__(self, K_s=0.0):

		self.K_s = K_s

	def printSelf(self):

		print "Motor:"
		print "K_s = ", self.K, "pN / nm"

		print "L_eq = ", self.lo_eq, "nm"
		print "Bound = ", self.bound 

	def bind(self, l_o):

		if(self.bidirectional):
			if(np.random.rand() > 0.5):
				self.sign = 1
			else:
				self.sign = -1

		#reset overlap
		self.lo_eq = l_o
		self.lo_eq_0 = l_o
		self.bound = True

	def unbind(self, l_o):

		self.bound = False
		self.sign = 1

	def setAsBidirectional(self):

		self.bidirectional = True

	#Update equilibrium length according to the relation:
	#	lo_eq += step_size
	#	
	#when the motor reaches lo_eq equal to a single filament length, 
	#the walking is stopped. In simulation, this should not be reached
	#but it is just a safeguard.	
	#
	def walk(self, filamentLength):

		self.lo_eq = min(filamentLength, self.lo_eq + self.sign * self.step_size)

	#Calculate stretching force given a filament overlap.
	#
	def calculateForce(self, l_o):

		if(self.bound):
			force = self.K * (l_o - self.lo_eq)
			return force
		else:
			return 0.0

	#Calculate stretching energy given a filament overlap
	#
	def calculateEnergy(self, l_o):

		if(self.bound):
			energy = self.K * (l_o - self.lo_eq)**2 * 0.5
			return energy
		else:
			return 0.0

	#reset motor
	def reinitialize(self):

		self.lo_eq = 0
		self.bound = False




