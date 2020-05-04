# -*- coding: utf-8 -*-
"""

Example : frequency linear sweep 

"""
#setup the environment
#
import sys
import numpy as np
import matplotlib.pyplot as plt
myModulePath = 'C:\\Users\\ricca\\.spyder-py3'
sys.path.append( myModulePath )
import hilbHuangEMD
#
# define the signal
#
acqFrq = 1e3
samplTime = 1.0/acqFrq
sigLen = 13000
xx = np.arange( 0, sigLen )
timeA = samplTime*xx
#
# linearly varying frequency
varLinA = 6 + 1.2 * timeA
#
SinA = np.sin( 2*np.pi*varLinA*timeA )
#
# now get the Hilbet Transform and the corresponding estimated phase
#
# the function returns also a 2nd phase, calculated limiting the minimum
# absolute values of the original signal to avoid division by zero
#
[ hlbTrsfSinA, hlbPhsSinA, hilbAltPhsA ] = hilbHuangEMD.hilbTransf( timeA, SinA, acqFrq )
C:\Users\ricca\.spyder-py3\hilbHuangEMD.py:189: RuntimeWarning: divide by zero encountered in true_divide
  hilbPhsOut = np.arctan( np.real( hilbSignOut ) / signalIn )
#
# unwrap the phases  
linPhsSinA = hilbHuangEMD.phaseUnwrap( hlbPhsSinA )
linPhsAltA = hilbHuangEMD.phaseUnwrap( hilbAltPhsA )
#
# the minimum values show a pi/2 negative offset 
min( linPhsSinA )
Out[548]: -1.5707963267948966
#
np.pi*0.5
Out[549]: 1.5707963267948966
#
# instantaneaous frequency
instFrqSinA = hilbHuangEMD.hilbInstFrq( acqFrq, linPhsSinA )
instFrqAltA = hilbHuangEMD.hilbInstFrq( acqFrq, linPhsAltA )
#
# the linear relationship emerges when the estimated frequencies are plotted
# this is the derivative of the phase 
#
# Assuming a sine-wave signal, a polynomial fit will recover the original 
# frequency sweep
#
from numpy.polynomial.polynomial import polyfit
#
# fit a 3rd order polynomial
# correct the  phase offset and rescale by 1/(2*pi)
[ coef, stats ] = polyfit( timeA, 0.5*( 0.5*np.pi+ linPhsSinA )/np.pi, 3, full=True )
[ coef1, stats1 ] = polyfit( timeA, 0.5*( 0.5*np.pi+ linPhsAltA )/np.pi, 3, full=True )
#
# the coefficient arrays return the original linear function
#
coef
Out[564]: array([ 2.38910502e-03,  5.99796444e+00,  1.20041453e+00, -2.31560607e-05])
coef1
Out[565]: array([ 2.38910502e-03,  5.99796444e+00,  1.20041453e+00, -2.31560607e-05])

