# -*- coding: utf-8 -*-
"""

quick loop script

"""
import sys
import numpy as np
myModulePath = 'C:\\Users\\ricca\\.spyder-py3'
# 
if ( myModulePath not in sys.path ):
    sys.path.append( myModulePath )
#   
import hilbHuangEMD
#
#
loopMaxIter = 40
rmsMinDiff = 1.0e-3
#
acqFrq = 1e3
samplTime = 1.0/acqFrq
sigLen = 13000
xx = np.arange( 0, sigLen )
siftCnvLim = 1e-8
siftMaxIter = int( 40e3 )
#
# auxiliary variables
timeAx = samplTime*xx
varLinA = 6 + 1.2 * timeAx
rampA = 3 * timeAx
rampB = -4.0 * ( timeAx > 7 ) * ( timeAx - 7 )
rampC = 2.0 * ( timeAx > 10 ) * ( timeAx - 10 )
varRamp = rampA + rampB + rampC
#
# signal
signal = 1.7 * np.cos( 2*np.pi*2*timeAx ) + 3.2 * np.sin( 2*np.pi*varLinA*timeAx ) + 3 * np.random.random( sigLen ) + 0.6*varRamp
#
rmsSign = np.sqrt( sum( ( signal - np.mean( signal ) )**2 ) / sigLen )
sigFFT = np.fft.fft( signal - np.mean(signal) ) / sigLen
#
#[ pchpMode, ncnts ] = hilbHuangEMD.extractMode( timeAx, signal, cnvLim, 40e3, hilbHuangEMD.siftStepPchp )
#
pchpMode = 0.0 * timeAx
modeFFT = 0.0*sigFFT
#
# arrays initialisation
signalArray = [ signal.copy() ]
modeArray = [ pchpMode.copy() ]
rmsModes = [ 0 ]
rmsSignals = [ rmsSign ]
rmsDiffs = [ 0 ]
fftSignals = [ sigFFT.copy() ]
fftModes = [ modeFFT.copy() ]
siftCounts = [ 0 ]
#
nIter = 0
while True:
    # loop operations
    [ pchpMode, ncnts ] = hilbHuangEMD.extractMode( timeAx, signal, siftCnvLim, siftMaxIter, hilbHuangEMD.siftStepPchp )
    #
    siftCounts = np.append( siftCounts, [ ncnts ] )
    signal -= pchpMode
    #
    # rms
    rmsSign = np.sqrt( sum( ( signal - np.mean( signal ) )**2 ) / sigLen )
    rmsMode = np.sqrt( sum( ( pchpMode - np.mean( pchpMode ) )**2 ) / sigLen )
    #
    # fft
    sigFFT = np.fft.fft( signal - np.mean( signal ) )/sigLen
    modeFFT = np.fft.fft( pchpMode - np.mean( pchpMode ) )/sigLen
    # 
    modeArray = np.append( modeArray, [ pchpMode ], 0 )
    signalArray = np.append( signalArray, [ signal ], 0 )
    #
    #
    rmsModes = np.append( rmsModes, [ rmsMode ] )
    rmsSignals = np.append( rmsSignals, [ rmsSign ] ) 
    #
    #
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
    if ( abs( rmsDiff ) < rmsMinDiff ) | ( nIter > loopMaxIter ) :
        break
    
