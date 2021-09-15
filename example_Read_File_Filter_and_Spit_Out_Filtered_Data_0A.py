# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:31:34 2021

@author: riccardo.carrozzo
"""
#
# example of filtered data 
# 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# Using an EMD loop to remove high frequency components from the orginal signal
# 
# NOTE : code is messy - used in debug/development - check comments
# 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# load environment: system, NumPy, Hilbert-Huang, etc.
#
import sys
import numpy as np
myModulePath = 'C:\\RC_Work\\GitHubWork\\rcHHT\\devel'
if ( myModulePath not in sys.path ):
    sys.path.append( myModulePath )
import hilbHuangEMD
import matplotlib.pyplot as plt
import h5py
#
#
# cd to data folder
# cd C:\RC_Work\Matlab_Work\SandBox\Pd18_Pd16_ThermalModel_TestData\Danfoss_Other\Danfoss_Dyno_MTL-402\Test 1\Coolant Flow U-W 6.0lmin\100 rpm 500 Nm Brak U-W Coolant\TC2D results
#
# cd C:\RC_Work\Matlab_Work\SandBox\Pd18_Pd16_ThermalModel_TestData\Danfoss_Other\Danfoss_Dyno_MTL-402\Test 1\Coolant Flow U-W 6.5lmin\100 rpm 1250 Nm Brak 6.5lmin\TC2D results
cd C:\RC_Work\Matlab_Work\SandBox\Pd18_Pd16_ThermalModel_TestData\Danfoss_Other\Danfoss_Dyno_MTL-402\Test 1\Coolant Flow U-W 6.0lmin\100 rpm 100 Nm Mot U-W Coolant\TC2D results
#
# 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# few options to grab input data from text files:
# - using "readlines" -> slightly more flexible
# - using "genfromtext" -> need to specify how the data is laid out
# 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# helper function to get float data
def chkFloat(x):
    try:
        float(x)
        return True
    except:
        return False
#    
# ------------------------------------------------------------------------
# NOTE:
#       2 ways of getting the data out
# ------------------------------------------------------------------------
#
# 1) get the text data lines - this requires more coding but is also more robust
with open('dtcW_NM1250pos_WarmUp_OvSmpl.txt') as filId:
    listData = filId.readlines()
#
# lines read 
nLines = len(listData)
#
# init the float array
fltData = np.array( [ float(xx) for xx in listData[0].split(' ') if chkFloat(xx) ] )
# 
# get float data in temp storage
for ii in range(1, nLines):
    tmpData = np.array([float(xx) for xx in listData[ii].split(' ') if chkFloat(xx)])
    fltData = np.append(fltData, tmpData)
#
# NOTE: number of lines of input text file is not number of data lines
fltLen   = len( fltData )/2
fltData2 = fltData.reshape(fltLen,2)
#
timeIn = fltData2[:,0].copy()
signal = fltData2[:,1].copy()
#
# ------------------------------------------------------------------------
# 2) use genfromtxt - need to specify header/footer lines to skip, etc.
# dataList = np.genfromtxt( "dtcU_Up_100rpm_NM0500pos_0_OvSmpl.txt", usecols=(0,1), dtype="float", skip_header=3 )
dataList = np.genfromtxt( "dtcU_NM0100pos_WarmUp_0_OvSmpl.txt", usecols=(0,1), dtype="float", skip_header=3 )
#
timeIn = dataList[:,0].copy()
signal = dataList[:,1].copy()
#
# ------------------------------------------------------------------------
# filtering loop initialisation
#
# the filering process aims at conserving the low-frequency RMS minimising 
# any phase distortion / response delay 
# 
# use a number of loop-stopping criteria
#
# ----------------------------------------
# Sifting 
siftCnvLim = 5e-4
siftMaxIter = int( 40e3 )
siftCounts = [ 0 ]
# ----------------------------------------
#
loopMaxIter = 200
rmsMinDiff = 1.0e-3
nIter = 0
#
# initialise various buffers
# - extracted modes
# - FFT spectra
# - RMS
# - log grad
# ... etc.
#
sigLen = len( signal )
rmsSign = np.sqrt( sum( ( signal - np.mean( signal ) )**2 ) / sigLen )
sigFFT = np.fft.fft( signal - np.mean(signal) ) / sigLen
modeFFT = 0.0*sigFFT
pchpMode = 0.0*modeFFT.real
signalArray = [ signal.copy() ]
modeArray = [ pchpMode.copy() ]
rmsModes = [ 0 ]
rmsSignals = [ rmsSign ]
rmsDiffs = [ 0 ]
rmsGrad = [0]
fftSignals = [ sigFFT.copy() ]
fftModes = [ modeFFT.copy() ]
#
# mode rms min/max flags
#
rmsModeMin = False
rmsModeMax = False
rmsRatioUp = False
rmsRatioLastUp = True
#
# Filtering Loop
#
while True:
    #
    # Sifting
    [ pchpMode, ncnts ] = hilbHuangEMD.extractMode( timeIn, signal, siftCnvLim, siftMaxIter, hilbHuangEMD.siftStepPchp )
    #
    # store results and set signal for next iteration
    siftCounts = np.append( siftCounts, [ ncnts ] )
    signal -= pchpMode
    #
    # rms calc
    rmsSign = np.sqrt( sum( ( signal - np.mean( signal ) )**2 ) / sigLen )
    rmsMode = np.sqrt( sum( ( pchpMode - np.mean( pchpMode ) )**2 ) / sigLen )
    #
    # fft calc
    sigFFT = np.fft.fft( signal - np.mean( signal ) )/sigLen
    modeFFT = np.fft.fft( pchpMode - np.mean( pchpMode ) )/sigLen
    # 
    # signal arrays
    modeArray = np.append( modeArray, [ pchpMode ], 0 )
    signalArray = np.append( signalArray, [ signal ], 0 )
    #
    # rms arrays
    rmsModes = np.append( rmsModes, [ rmsMode ] )
    rmsSignals = np.append( rmsSignals, [ rmsSign ] ) 
    #
    # rms Gradient - this is actually a Log Gradient - d(RMS)/RMS
    # used for loop-stopping and to refine sifting limit
    #
    if ( nIter < 6 ):
        #
        # zero rms gradient
        rmsGrad = np.append( rmsGrad, [0] )
    else:
        #
        # compute gradient (z-5)
        rmsGrad = np.append( rmsGrad, [ ( rmsSign - rmsSignals[nIter-5] )/rmsSign ] )
    # 
    # check sifting limit
    if ( all( rmsGrad[nIter-5:nIter] < 0 ) ):
        if ( all( rmsGrad[nIter-5:nIter] > -1e-2 ) ):
            #
            # flattening -> change Sifting limit
            siftCnvLim = 0.9*siftCnvLim
    #
    # FFT arrays
    fftSignals = np.append( fftSignals, [ sigFFT ], 0 )
    fftModes = np.append( fftModes, [ modeFFT ], 0 )
    #
    # iteration counter
    nIter += 1
    #
    #
    endIdx = len( rmsSignals ) - 1
    rmsDiff = rmsSignals[ endIdx-1 ] - rmsSignals[ endIdx ]
    rmsDiffs = np.append( rmsDiffs, [ rmsDiff ] )
    # 
    # set rmw flags
    rmsRatioUp = ( rmsMode/rmsSign > rmsModes[nIter-1]/rmsSignals[nIter-1] )        
    #
    # loop exit condition - not used any longer - too simplstic
    #
    # ---------------------------------------------------------------
    # rmsDiff < minDiff limit OR not converged 
    # if ( abs( rmsDiff ) < rmsMinDiff ) | ( nIter > loopMaxIter ) :
    #    break
    #
    # ---------------------------------------------------------------
    # 
    # Loop Breakers
    # 
    # RMS of stripped modes rising with RMS - OR - max iteration limit reached
    if ( ( rmsModeMax & rmsRatioUp  & ( rmsSign < 0.3*rmsSignals[0] ) ) |  ( nIter > loopMaxIter ) ) : 
        #
        # seen maximum and rising again OR not converged
        break
    #
    # 30 iterations and long sifting loops
    if ( ( nIter > 30 ) & ( all( siftCounts[nIter-5:nIter] > 10000 ) ) ):
        break
    #
    # 100 iterations AND nearly-flat rms gradient
    if ( ( nIter > 100 ) & (all( abs( rmsGrad[nIter-20:nIter] ) < 1e-4 ) ) ):
        break
    #
    # set max flag
    if ( ~rmsRatioUp & rmsRatioLastUp ) :
        # 1st max achieved
        rmsModeMax = True
    #    
    rmsRatioLastUp = rmsRatioUp

#
# Matplotlib plots
#
fig1 = plt.figure(1)
# original signal
plt.plot( timeIn, signalArray[0,], '.-.', label=('Original') )
# filtered signal
plt.plot( timeIn, signal, '.:', label=('Filtered') )
#
# Fourier Spectrum
acqFrq = 10.0
frqAx = 0.5*acqFrq*np.arange(0, int(0.5*sigLen) )/(0.5*sigLen)
plt.plot( frqAx, 2*abs( fftSignals[0,0:int(0.5*sigLen)]), '.', frqAx, 2*abs( sigFFT[0:int(0.5*sigLen)]), '.:' ) 
#
# .....................................................
# Output
#
# not too bad - spit it out
txtDataOut = [ np.transpose( timeIn.copy() ) ]
txtDataOut = np.append( txtDataOut, [ np.transpose( signal ) ], 0 )
#
# Write to file and clean up
# 
np.savetxt( 'dtcU_NM0100pos_WarmUp_Fild_01.txt', np.transpose( txtDataOut ) )
del txtDataOut
del dataList