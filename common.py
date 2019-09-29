
import math
import numpy as np

from scipy.misc import factorial

#	This module contains global functions and constants used in the simulator.
#
kT   =  4.1   #thermal energy (pN * nm)
fric_coeff   = 1E-3    #fric coefficient used (pN * s / nm)
actin_mass   = 6.97E-20 #mass of actin monomer (pN * s^2 / nm)
actin_length = 2.7 		#length of actin monomer (nm)

#from stackoverflow:
#http://stackoverflow.com/questions/25828184/fitting-to-poisson-histogram
def poisson(k, lamb):
	return (lamb**k/factorial(k)) * np.exp(-lamb)

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def negLogLikelihood(params, data):
	lnl = - np.sum(np.log(poisson(data, params[0])))
	return lnl

def asymp(x, a, b):
	return a * (1 - np.exp(-x / b))