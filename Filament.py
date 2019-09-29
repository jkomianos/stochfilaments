
import common

import math
import numpy as np

#	This class represents a 1D filament.
#	Contains functions to update and track position via Langevin dynamics.
#
class Filament:

	#MEMBER VARIABLES

	v = 0			#Velocity if doing standard langevin (nm / s)
	mass = 0		#Mass of filament (pN * s^2 / nm)

	mp  = 0.0  		#Midpoint of filament (nm)
	omp = 0.0       #Original midpoint (nm)
	L   = 0.0 		#Filament length (nm)


	eqmp = 0.0 		  #Equilibrium midpoint (nm)
	K  	 = 0.0        #Spring constant (pN / nm)

	#other constants
	sigma = 6         #Width of actin filament (nm)

	#	Initializes filament and sets all relevant parameters
	#
	def __init__(self, mp, L, K=0.0):

		self.mp = mp
		self.L  = L
		self.K  = K

		#set eq midpoint and original to initial
		self.eqmp = self.omp = mp

		#initialize mass
		self.calcMass()

	#	Reset the midpoint to the original position (omp)
	def resetPosition(self):

		self.mp = self.omp 


	#   Prints all relevant object variables
	def printSelf(self):
		
		print "Midpoint = ", self.mp, " nm"
		print "Equilibrium midpoint = ", self.eqmp, " nm"
		print "Length = ", self.L, " nm"
		print "Spring constant = ", self.K, " pN / nm"


	# 	Calculates mass of filament simply by:
	#
	#	mass = (L / actin_length) * actin_mass
	#
	#	where:
	#	l_mon = length of actin monomer
	#	m_mon = mass of actin monomer 
	#
	def calcMass(self):

		self.mass = (self.L / common.actin_length) * common.actin_mass


