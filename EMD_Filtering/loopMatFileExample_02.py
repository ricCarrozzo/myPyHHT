# -*- coding: utf-8 -*-
"""
Created on Thu May 20 18:04:15 2021

@author: ricca

Example reading a .mat file

"""
import sys
import numpy as np
myModulePath = 'C:\\RC_Work\\GitHubWork\\rcHHT\\devel'
# myModulePath = 'C:\\Users\\ricca\\.spyder-py3'
# 
if ( myModulePath not in sys.path ):
    sys.path.append( myModulePath )
#   
import hilbHuangEMD
# plot stuff
import matplotlib.pyplot as plt
import h5py
#
# fil = h5py.File( 'dataCW0200v7_3.mat', 'r' )
# tst01 = fil.get( 'igbtTempCW0200' )
# tim01 = fil.get('acqTimeCW0200' )
# timeIn = np.array( tim01 ).flatten()
# igbtIn = np.array( tst01 ).flatten()
#
# cd to data folder of interest
# cd C:\RC_Work\Matlab_Work\SandBox\Pd18_Pd16_ThermalModel_TestData\Danfoss_Other\Danfoss_Dyno_MTL-402\Test 5\T5_150edeg\PhaCur 450A
cd C:\RC_Work\Matlab_Work\SandBox\Pd18_Pd16_ThermalModel_TestData\Danfoss_Other\Danfoss_Dyno_MTL-402\Test 6\T6_30edeg_PWM0\PhaCur 420A_1
#
# example of data import for cell array types from .mat v7.3 files
#
# file object
#
# dataFile = h5py.File( 'dtcDataOut.mat', 'r' )
# dataFile = h5py.File( 'T6_Dtc_CoolOut_00.mat', 'r' )
# keys -> variables in data file
# dataFile.keys()
#
# cell array for acquisition time
# timeRef = dataFile['acqTime450A_addDt']
# properties
# dir( timeIn )
# text fields for var name
# timeIn.name
# this is the reference/handle/pointer of the data array
# dataFile[ timeIn[0][0] ]
# this formats the 2D data array (1,<NumOfPoints>) to 1D  
#timeData = np.array( dataFile[ timeIn[0][0] ] ).flatten()
#
# the temperature data 
# dtcUin = dataFile[ 'dtcU_450A_addDt' ]
# in this case, this is the reference -> grab the data
# ddtcUData = np.array( dtcUin ).flatten()
#
# check it out
# plt.plot( timeData, dtcUData, '.:' )
#
# 1D cell array and 2D cell array 
timeRefDtc = dataFile['acqTime420A_addDt']
timeRefCool = dataFile['acqTimeNM_ctlDt']
#
# 2D cell array
timeDataCoolObj = dataFile[ timeRefCool[0][0] ][0,0]
#
# the Coolant Temperature time axis
timeDataCool = np.array( dataFile[ timeDataCoolObj ] ).flatten()
#
# teh DTC Temperature time axis
timeDataDtc = np.array( dataFile[ timeRefDtc[0][0] ] ).flatten()
#
# the temperature data 
tempRefCool = dataFile[ 'coolOutTempNM_ctlDt' ]
tempDtcVRef = dataFile[ 'dtcV_420A_addDt' ]
#
# the Coolant Temperature data
coolTData = np.array( dataFile[ tempRefCool[0][0] ] ).flatten()
#
# in this case, this is the reference -> grab the data
dtcVData = np.array( tempDtcVRef ).flatten()
# check it out
plt.plot( timeData, dtcUData, '.:' )
#
# Initialise Loop parameters
loopMaxIter = 40
rmsMinDiff = 1.0e-3
siftCnvLim = 1e-8
siftMaxIter = int( 40e3 )
#
# populate "signal" and "timeIn"
# signal = igbtIn.copy()
signal = coolTData.copy()
timeIn = timeDataCool.copy()
# carry on with loop arrays initialisation
sigLen = len( signal )
rmsSign = np.sqrt( sum( ( signal - np.mean( signal ) )**2 ) / sigLen )
sigFFT = np.fft.fft( signal - np.mean(signal) ) / sigLen
modeFFT = 0.0*sigFFT
pchpMode = 0.0*modeFFT.real
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
    [ pchpMode, ncnts ] = hilbHuangEMD.extractMode( timeIn, signal, siftCnvLim, siftMaxIter, hilbHuangEMD.siftStepPchp )
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
    
#
# save data out 
#
# 2 arrays -> 1 matrix
#
# temperature
coolOutT = np.transpose( signalArray[5,].copy() )
# time in 1st
txtDataOut = [ np.transpose( timeIn.copy() ) ]
# append tenperature
txtDataOut = np.append( txtDataOut, [ np.transpose( coolOutT ) ], 0 )
# transpose for output as 2-columns 
np.savetxt( 'coolOutData.txt', np.transpose( txtDataOut ) )

