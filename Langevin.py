
import common

import math
import numpy as np

from Filament import Filament
from Motor import Motor

#	This class is a langevin dynamics integrator for the two-filament or single filament system.
#
class Langevin:

	# 	Compute the instantaneous overlap between the right and left filaments as:
	#	
	#	l = (mp_l + L_l/2) - (mp_r - L_r/2)
	#
	#	where:
	#	mp = midpoints of left and right filaments, respectively.
	#	
	def computeOverlap(self, l_fil, r_fil):

		return (l_fil.mp + l_fil.L) - (r_fil.mp - r_fil.L)

	#	Integrates, for two filaments, the overdamped Langevin equation for numsteps:
	#
	#	dx = ((K * (x - x0) + F_motor / 2) / fric_coeff) dt + sqrt(2 kT dt / fric_coeff) * W(1)
	#
	#	where:
	#	W(1) = a wiener process, normally distributed random variable N(0,1)
	#
	#	Returns true if successful, and false otherwise. We assume that the motor ensemble
	#	distributes its force equally among the two filaments, hence the F_motor/2.
	#
	def overdampedLangevin(self, l_fil, r_fil, motor, timestep, numsteps): 

		for step in xrange(numsteps):

			#Calculate motor force
			F_motor = motor.calculateForce(self.computeOverlap(l_fil, r_fil))

			#Left filament.  Note that the sign of F_motor is reversed
			ldmp = (-l_fil.K * (l_fil.mp - l_fil.eqmp) - F_motor) * timestep / common.fric_coeff \
				  + math.sqrt(2 * common.kT * timestep / common.fric_coeff) \
				  * np.random.normal(0,1)

			l_fil.mp += ldmp

			#Right filament.
			rdmp = (-r_fil.K * (r_fil.mp - r_fil.eqmp) + F_motor) * timestep / common.fric_coeff \
				  + math.sqrt(2 * common.kT * timestep / common.fric_coeff) \
				  * np.random.normal(0,1)

			r_fil.mp += rdmp

		#sanity check
		if(l_fil.mp != l_fil.mp or l_fil.mp == float('Inf') or l_fil.mp == -float('Inf') or
		   r_fil.mp != r_fil.mp or r_fil.mp == float('Inf') or r_fil.mp == -float('Inf')): 

			print "Produced garbage values in langevin dynamics. Try adjusting timestep."

			#reset if reuse is required
			l_fil.mp = l_fil.omp
			r_fil.mp = r_fil.omp

			return False

		else: return True

	# 	Integrates, for a single filament, the overdamped Langevin eq as below.
	#	Note that this integration contains no motor forces
	def overdampedLangevinSingle(self, fil, timestep, numsteps):

		for step in xrange(numsteps):

			#Left filament.  Note that the sign of F_motor is reversed
			dmp = -(fil.K * (fil.mp - fil.eqmp)) * timestep / common.fric_coeff \
				  + math.sqrt(2 * common.kT * timestep / common.fric_coeff) \
				  * np.random.normal(0,1)

			fil.mp += dmp

		#sanity check
		if(fil.mp != fil.mp or fil.mp == float('Inf') or fil.mp == -float('Inf')): 

			print "Produced garbage values in langevin dynamics. Try adjusting timestep."

			#reset if reuse is required
			fil.mp = fil.omp

			return False

		else: return True


	#	Integrates, for two filaments the standard Langevin equation for numsteps:
	#
	#	dv  = - (fric_coeff * v / m) * dt + ((K * (x - x0) + F_motor / 2)/ m) * dt + 
	#		  	(sqrt(2 * kT * fric_coeff * dt) / m) * W(1)
	#
	#	where:
	#	W(1) = a wiener process, normally distributed random variable N(0,1)
	#	m = mass of actin filament
	#	
	#	Returns true if successful, and false otherwise. We assume that the motor ensemble
	#	distributes its force equally among the two filaments, hence the F_motor/2.
	#
	def standardLangevin(self, l_fil, r_fil, motor, timestep, numsteps):

		#Assuming start with zero velocity
		l_fil.v = 0
		r_fil.v = 0

		for step in xrange(numsteps):

			#Calculate motor force
			F_motor = motor.calculateForce(self.computeOverlap(l_fil, r_fil))

			#Left filament.  Note that F_motor is reversed
			dv = - common.fric_coeff * l_fil.v * timestep / l_fil.mass \
				 - (l_fil.K * (l_fil.mp - l_fil.eqmp) + F_motor) * timestep / l_fil.mass \
				 + math.sqrt(2 * common.kT * common.fric_coeff * timestep) \
				 * np.random.normal(0,1) / l_fil.mass

			l_fil.v += dv
			l_fil.mp += l_fil.v * timestep

			#Right filament.
			dv = - common.fric_coeff * r_fil.v * timestep / r_fil.mass \
				 - (r_fil.K * (r_fil.mp - r_fil.eqmp) - F_motor) * timestep / r_fil.mass \
				 + math.sqrt(2 * common.kT * common.fric_coeff * timestep) \
				 * np.random.normal(0,1) / r_fil.mass

			r_fil.v += dv
			r_fil.mp += r_fil.v * timestep

		#sanity check
		if(l_fil.mp != l_fil.mp or l_fil.mp == float('Inf') or l_fil.mp == -float('Inf') or
		   r_fil.mp != r_fil.mp or r_fil.mp == float('Inf') or r_fil.mp == -float('Inf')): 

			print "Produced garbage values in langevin dynamics. Try adjusting timestep."

			#reset if reuse is required
			l_fil.mp = l_fil.omp
			r_fil.mp = r_fil.omp

			return False

		else: return True

	# 	Integrates, for a single filament, the standard Langevin eq as below.
	#	Note that this integration contains no motor forces
	def standardLangevinSingle(self, fil, timestep, numsteps):

		for step in xrange(numsteps):

			#Left filament.  Note that the sign of F_motor is reversed
			dv = - common.fric_coeff * fil.v * timestep / fil.mass \
				 - (fil.K * (fil.mp - fil.eqmp)) * timestep / fil.mass \
				 + math.sqrt(2 * common.kT * common.fric_coeff * timestep) \
				 * np.random.normal(0,1) / fil.mass

			fil.v += dv
			fil.mp += fil.v * timestep

		#sanity check
		if(fil.mp != fil.mp or fil.mp == float('Inf') or fil.mp == -float('Inf')): 

			print "Produced garbage values in langevin dynamics. Try adjusting timestep."

			#reset if reuse is required
			fil.mp = fil.omp

			return False

		else: return True

