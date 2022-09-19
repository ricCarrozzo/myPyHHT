# -*- coding: utf-8 -*-
#
# Simple implementation of Hilbert-Huang transform
#
"""
Module for Empirical Mode Decomposition and instantaneous frequency estimation 
of non-linear transient signals.

The module includes the following functions:
    - extractMode : returns the 1st empirical mode identified from the 
                    input time-history, to the defined convergence limit
    - siftStep    : sifting function returning the candidate mode signal
                    at the current iteration step of the EMD process
    - hilbTransf  : calculate the Hilbert Transform and the associated 
                    discontinuous phase signal
    - phaseUnwrap : takes a mod(pi) discontinuous phase signal and returns
                    the unwrapped continuous version
    - hilbInstFrq : Estimates the instantaneous frequency based on the Hilbert 
                    transform phase signal
"""
#
# Numpy and Scipy Interpolation required
import numpy as np
from scipy import interpolate as intp
#
# 
def siftStepCubSpl( xArray, sigArray ):
    """ EMD sifting function  : 
            - Input     -> [ xArray, sigArray ] signal time-history
            - Output    -> [ candidate mode, inf envelope, sup envelope ]
        calculates a candidate Empirical Mode as a point-to-point average of
        sup and inf envelopes - using Cubic Splines.
    """
    # 
    # signal length
    sigLen = len( sigArray )
    # point-to-point gradient signature - positive
    grdPls = 0 * sigArray
    grdPls[1:sigLen] = sigArray[1:sigLen] > sigArray[0:sigLen-1]
    # complementary - negative
    grdMns = grdPls < 1
    #
    # calculate indexes to local sup and inf elements
    #
    # negative and positive gradients
    negGrdId = 1*grdMns
    posGrdId = 1*grdPls
    # markers of sup and inf elements
    supMrkFlg = 0*grdPls
    infMrkFlg = 0*grdMns
    supMrkFlg[0:sigLen-1] = posGrdId[0:sigLen-1] + negGrdId[1:sigLen]
    infMrkFlg[0:sigLen-1] = negGrdId[0:sigLen-1] + posGrdId[1:sigLen]
    #
    # inf and sup indexes
    supIdx = supMrkFlg == 2
    infIdx = infMrkFlg == 2
    #
    # extend supIdx and infIdx to extremes
    supIdx[0] = True
    supIdx[sigLen-1] = True
    infIdx[0] = True
    infIdx[sigLen-1] = True
    #
    # calculate sup and inf envelopes by cubic spline
    # 
    infIntpFcn = intp.CubicSpline( xArray[infIdx], sigArray[infIdx] )
    supIntpFcn = intp.CubicSpline( xArray[supIdx], sigArray[supIdx] )
    #
    infEnv = infIntpFcn( xArray )
    supEnv = supIntpFcn( xArray )
    #
    # calculate candidate mode
    cndMode = sigArray - 0.5*( infEnv + supEnv )
    #
    return [ cndMode, infEnv, supEnv ]
    #
    #
def siftStepPchp( xArray, sigArray ):
    """ EMD sifting function  : 
            - Input     -> [ xArray, sigArray ] signal time-history
            - Output    -> [ candidate mode, inf envelope, sup envelope ]
        calculates a candidate Empirical Mode as a point-to-point average of
        sup and inf envelopes - using PCHIP interpolation.
    """
    # 
    # signal length
    sigLen = len( sigArray )
    # point-to-point gradient signature - positive
    grdPls = 0 * sigArray
    grdPls[1:sigLen] = sigArray[1:sigLen] > sigArray[0:sigLen-1]
    # complementary - negative
    grdMns = grdPls < 1
    #
    # calculate indexes to local sup and inf elements
    #
    # negative and positive gradients
    negGrdId = 1*grdMns
    posGrdId = 1*grdPls
    # markers of sup and inf elements
    supMrkFlg = 0*grdPls
    infMrkFlg = 0*grdMns
    supMrkFlg[0:sigLen-1] = posGrdId[0:sigLen-1] + negGrdId[1:sigLen]
    infMrkFlg[0:sigLen-1] = negGrdId[0:sigLen-1] + posGrdId[1:sigLen]
    #
    # inf and sup indexes
    supIdx = supMrkFlg == 2
    infIdx = infMrkFlg == 2
    #
    # extend supIdx and infIdx to extremes
    supIdx[0] = True
    supIdx[sigLen-1] = True
    infIdx[0] = True
    infIdx[sigLen-1] = True
    #
    # calculate sup and inf envelopes by cubic spline
    # 
    infIntpFcn = intp.PchipInterpolator( xArray[infIdx], sigArray[infIdx] )
    supIntpFcn = intp.PchipInterpolator( xArray[supIdx], sigArray[supIdx] )
    #
    infEnv = infIntpFcn( xArray )
    supEnv = supIntpFcn( xArray )
    #
    # calculate candidate mode
    cndMode = sigArray - 0.5*( infEnv + supEnv )
    #
    return [ cndMode, infEnv, supEnv ]
    #
    #
def extractMode( timeIn, signalIn, cnvgLim, maxIter, getMode ):
    """ EMD mode extraction function :
                - Input     : [timeIn, signalIn ] -> signal time-history
                              cnvgLim             -> convergence limit
                              maxIter             -> max iterations limit
                              getMode             -> sifting function
                - Output    : empirical mode
                              number of iterations to achieve convergence
    """
    #
    #initialise convergence limit and sifting input 
    iterLim = 1e5*cnvgLim
    siftInput = signalIn
    nIter = 0
    #
    while ( ( iterLim > cnvgLim ) and ( nIter <= maxIter ) ) :
        # calculate candidate mode
        [ siftOutput, dwnenv, upenv ] = getMode( timeIn, siftInput )
        #
        iterLim = sum( ( abs( siftInput - siftOutput ) )**2 ) / sum( siftInput**2 )
        #
        siftInput = siftOutput
        nIter +=1
        
    return [ siftOutput, nIter ]
#
#
#
def hilbTransf( timeIn, signalIn, acqFreq ):
    """ Hilbert Transform Function :  
            - Input     : [ timeIn, signalIn ]  -> signal time-history
                          acqFreq               -> acquisition frequency
            - Output    : hilbSignOut           -> Hilbert Transform (complex)
                          hilbPhsOut            -> Hilbert Phase mod(pi)
                          altPhsOut             -> Hilbert Phase mod(pi)
                                                   No Division by 0
        NOTE: timeIn assumed equally spaced by ( 1.0/acqFreq )
    """
    # function parameters
    sigLen = len( signalIn )
    #
    # take the fft of the signal to be transformed
    sigInFFT = np.fft.fft( signalIn - np.mean( signalIn ) ) / sigLen
    #
    # define the Hilbert Transform frequency axis - based on acqFreq
    # start from the standard FFT frequency array [ 0 acqFreq/2, -acqFreq/2 0 ]
    baseHlbFrq = 1 * np.arange( 0, sigLen )
    baseHlbFrq[ int( 0.5*sigLen ): ] -= sigLen
    #
    # Hilbert Transform frequency array
    hlbFrqAx = 0.5 * acqFreq * baseHlbFrq / sigLen
    #
    # get the Hilbert Transform array
    hilbSignOut = sigLen * np.fft.ifft( -1j * np.sign( hlbFrqAx ) * sigInFFT )
    # 
    # obviously, as FFT is a complex array, so Hilbert will be as well
    #
    # calculate the Hilbert Phase array
    hilbPhsOut = np.arctan( np.real( hilbSignOut ) / signalIn )
    #
    signalOp = signalIn
    idxSmallPls = ( signalIn >= 0.0 ) & ( signalIn < 1.0e-8 )
    idxSmallMns = ( signalIn < 0.0 ) & ( signalIn > -1.0e-8 )
    signalOp[ idxSmallPls ] = 1.0e-8
    signalOp[ idxSmallMns ] = -1.0e-8
    altPhsOut = np.arctan( np.real( hilbSignOut ) / signalOp )    
    #
    # Output arrays
    return [ hilbSignOut, hilbPhsOut, altPhsOut ]
#
#
def phaseUnwrap( rawPhaseIn ):    
    """ phaseUnwrap :   
            - Input     : rawPhaseIn  -> discontinuous mod(pi) phase signal
            - output    : phsToUnwrp  -> unwrapped continuous phase signal
    """
    #
    # initialise working variables and parameters
    sigLen = len( rawPhaseIn )
    phsIdxArray = np.arange( 0, sigLen )
    phsToUnwrp = rawPhaseIn
    #
    sigJmpMrkIdx = phsIdxArray < -1
    sigJmpMrkIdx[0:sigLen-1] = ( phsToUnwrp[0:sigLen-1] - phsToUnwrp[1:sigLen] ) > 2.8
    phsIdxJmp = phsIdxArray[ sigJmpMrkIdx ]
    #
    # loop on jump indexes and add (pi) from the discontinuity onwards
    for idx in phsIdxJmp :
        phsToUnwrp[ idx+1 : sigLen ] = np.pi + phsToUnwrp[ idx+1 : sigLen ]
         
    # output unwrapped phase array
    return phsToUnwrp
#
#
def hilbInstFrq( acqFrq, hilbEstimPhs ):
    """
    hilbInstFrq     : 
             - Input    : acqFreq       -> acquisition frequency
                          hilbEstimPhs  -> Hilbert instantaneous phase
             - Output   : hilbFrqOut    -> estimated instantaneous frequency 
    """
    # initialise estimated instantaneous frequency
    sigLen = len( hilbEstimPhs )
    hilbFrqOut = 0 * hilbEstimPhs
    # take numerical derivative
    hilbFrqOut[ 1:sigLen ] = hilbEstimPhs[ 1:sigLen ] - hilbEstimPhs[ 0:sigLen-1 ]
    # instantaneous frequency 
    hilbFrqOut = 0.5 / np.pi * acqFrq * hilbFrqOut
    #
    # done
    return  hilbFrqOut
#
# EOF
#

    

