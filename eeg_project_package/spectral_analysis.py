import os
import time
import numpy as np
import mne
#from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.stats import permutation_cluster_test
from sklearn.metrics import r2_score
from mne.datasets import somato
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import neurokit2 as nk
from scipy import signal,stats,fft
from spectrum import arburg,arma2psd,pburg
import statsmodels.regression.linear_model as transform
from scipy.signal import spectrogram

from numpy.fft import fft

from spectrum.correlation import CORRELATION
from spectrum.covar import arcovar, arcovar_marple
import spectrum.yulewalker as yulewalker
from spectrum.psd import ParametricSpectrum


def arma2psd_complex(A=None, B=None, rho=1., T=1., NFFT=4096, sides='default',
        norm=False):
    r"""Computes power spectral density given ARMA values.

    This function computes the power spectral density values
    given the ARMA parameters of an ARMA model. It assumes that
    the driving sequence is a white noise process of zero mean and
    variance :math:`\rho_w`. The sampling frequency and noise variance are
    used to scale the PSD output, which length is set by the user with the
    `NFFT` parameter.

    :param array A:   Array of AR parameters (complex or real)
    :param array B:   Array of MA parameters (complex or real)
    :param float rho: White noise variance to scale the returned PSD
    :param float T:   Sampling frequency in Hertz to scale the PSD.
    :param int NFFT:  Final size of the PSD
    :param str sides: Default PSD is two-sided, but sides can be set to centerdc.

    .. warning:: By convention, the AR or MA arrays does not contain the
        A0=1 value.

    If :attr:`B` is None, the model is a pure AR model. If :attr:`A` is None,
    the model is a pure MA model.

    :return: two-sided PSD

    .. rubric:: Details:

    AR case: the power spectral density is:

    .. math:: P_{ARMA}(f) = T \rho_w \left|\frac{B(f)}{A(f)}\right|^2

    where:

    .. math:: A(f) = 1 + \sum_{k=1}^q b(k) e^{-j2\pi fkT}
    .. math:: B(f) = 1 + \sum_{k=1}^p a(k) e^{-j2\pi fkT}

    .. rubric:: **Example:**

    .. plot::
        :width: 80%
        :include-source:

        import spectrum.arma
        from pylab import plot, log10, legend
        plot(10*log10(spectrum.arma.arma2psd([1,0.5],[0.5,0.5])), label='ARMA(2,2)')
        plot(10*log10(spectrum.arma.arma2psd([1,0.5],None)), label='AR(2)')
        plot(10*log10(spectrum.arma.arma2psd(None,[0.5,0.5])), label='MA(2)')
        legend()

    :References: [Marple]_
    """
    if NFFT is None:
        NFFT = 4096

    if A is None and B is None:
        raise ValueError("Either AR or MA model must be provided")

    psd = np.zeros(NFFT, dtype=complex)

    if A is not None:
        ip = len(A)
        den = np.zeros(NFFT, dtype=complex)
        den[0] = 1.+0j
        for k in range(0, ip):
            den[k+1] = A[k]
        denf = fft(den, NFFT)

    if B is not None:
        iq = len(B)
        num = np.zeros(NFFT, dtype=complex)
        num[0] = 1.+0j
        for k in range(0, iq):
            num[k+1] = B[k]
        numf = fft(num, NFFT)

    # Changed in version 0.6.9 (divided by T instead of multiply)
    if A is not None and B is not None:
        psd = rho / T * abs(numf)**2. / abs(denf)**2.
    elif A is not None:
        psd = rho / T / abs(denf)**2.
    elif B is not None:
        psd = rho / T * abs(numf)**2.

    return denf

def Power_burg_calculation_optimization(Epoch_compute,noverlap,N_FFT,f_max, n_per_seg,smoothing,filter_order):
    #burg = pburg(Epoch_compute,15,NFFT = nfft_test)
    data = Epoch_compute[:,:]
    xs, ys = [], []
    a = data.shape
    M = a[1]
    #print(M)
    L = n_per_seg
    #print(L)
    noverlap = noverlap
    LminusOverlap = L-noverlap
    #print(LminusOverlap)
    k = round((M-noverlap)/(L-noverlap))
    #print(k)
    xStart = np.array(range(0,k*LminusOverlap,LminusOverlap))
    #print(xStart)
    xEnd = []
    for i in xStart:
        xEnd.append(i+L-1)
    xEnd = np.array(xEnd)
    #print(xEnd)
    fres =f_max/N_FFT
    power = []
    Block_spectrum= []

    tab = np.array(range(k-1))
    trialspectrum = np.zeros([a[0],N_FFT])
    PSD_final = np.zeros([a[0],round(f_max/2)])
    Time_freq = np.zeros([a[0],len(tab),round(f_max/(2*fres))])
    time = np.linspace(0,round(M/f_max),k-1)
    for i in range(a[0]):
        Block_spectrum = []
        for numBlock in tab:
            aux_condition1data = Epoch_compute[i,xStart[numBlock]:xEnd[numBlock]]

            aux_condition1data = signal.detrend(aux_condition1data,type='constant')

            AR, sigma2 = transform.burg(aux_condition1data, filter_order)
            PSD = arma2psd(-AR,rho = sigma2,T = f_max, NFFT = N_FFT,sides='centerdc')
            # plt.plot(PSD)
            # plt.show()
            Block_spectrum.append(PSD)
            Time_freq[i,numBlock]=PSD[round(f_max/(2*fres)):round(f_max/fres)]
            block = np.mean(Block_spectrum,axis=0)
            trialspectrum[i] = np.array(block)

    return trialspectrum[:,round(f_max/(2*fres)):round(f_max/fres)],Time_freq,time


def Power_burg_calculation(Epoch_compute,noverlap,N_FFT,f_max, n_per_seg,filter_order):
    #burg = pburg(Epoch_compute,15,NFFT = nfft_test)

    a = Epoch_compute.shape
    print(a)
    M = a[2]  # M = trial length, in samples
    L = n_per_seg  # L = windowing size, in samples
    noverlap = noverlap  # size of overlapping segment btw windows, in samples
    k = round((M-noverlap)/(L-noverlap))  # nb of windows in the trial

    # Arrays of starting and ending indices, for cutting the signal in overlapping chunks
    xStart = np.array(range(0, k*(L-noverlap), L-noverlap))
    xEnd = xStart.copy() + L

    fres = f_max/N_FFT

    tab = np.array(range(k))  # tab = indices of overlapping windows
    trialspectrum = np.zeros([a[0],a[1],N_FFT])
    PSD_final = np.zeros([a[0],a[1],round(f_max/2)])
    Time_freq = np.zeros([a[0],a[1],len(tab),N_FFT//2])
    time = np.linspace(0,round(M/f_max),k-1)
    ITC = np.zeros((a[0], a[1], N_FFT),dtype=np.complex_)

    for i in range(a[0]):
        for j in range(a[1]):
            Block_spectrum = []
            Block_complex = []
            for numBlock in tab:
                windowData = Epoch_compute[i, j, xStart[numBlock]:xEnd[numBlock]]
                windowData = signal.detrend(windowData, type='constant')

                AR, sigma2 = transform.burg(windowData, filter_order)
                PSD = arma2psd(-AR, NFFT=N_FFT, sides='centerdc')
                print(PSD.shape)
                PSD_complex = arma2psd_complex(-AR, NFFT=N_FFT, sides='centerdc')
                Block_spectrum.append(PSD)
                Block_complex.append(PSD_complex)
                Time_freq[i, j, numBlock] = PSD[:PSD.shape[0]//2]
                # plt.plot(PSD)
                # plt.show()

            block = np.mean(Block_spectrum, axis=0)
            trialspectrum[i, j] = np.array(block)
            #print(np.angle(np.array(Block_complex).mean(0)))
            ITC[i, j] = np.exp(1j * np.angle(np.array(Block_complex).mean(0)))

    return trialspectrum[:,:,round(f_max/(2*fres)):round(f_max/fres)], Time_freq, time

def Power_calculation_welch_method(Epoch_compute,f_min,f_max,t_min,t_max,nfft,noverlap,nper_seg,pick,proje,averag,windowing, smoothing):
    #filtered = mne.filter.filter_data(Epoch_compute, 140, 7, 35)

    fres =f_max/nfft
    #psd_left,freqs_left = mne.time_frequency.psd_welch(Epoch_compute, fmin=f_min, fmax=f_max, tmin=t_min, tmax=t_max, n_fft=nfft, n_overlap=noverlap, n_per_seg=nper_seg, picks=pick, proj=proje, n_jobs=1, reject_by_annotation=True, average=averag, window=windowing, verbose=None)
    freqs_left,psd_left = signal.welch(Epoch_compute, fs=f_max, window=windowing, nperseg=nper_seg, noverlap=noverlap, nfft=nfft, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average=averag)
    b = psd_left.shape
    #print(freqs_left.shape[0])
    PSD_final = np.zeros([b[0],b[1],round(f_max/2)])
    if smoothing == True:
        for k in range(b[0]):
             for l in range(b[1]):
                 PSD_final[k,l,0] = (psd_left[k,l,0] + psd_left[k,l,1] +psd_left[k,l,2])/3
                 PSD_final[k,l,round(f_max/2)-1] = (psd_left[k,l,freqs_left.shape[0]-3] + psd_left[k,l,freqs_left.shape[0]-2] +psd_left[k,l,freqs_left.shape[0]-1])/3
                 for i in range(5,freqs_left.shape[0]-2,round(1/fres)):
                     PSD_final[k,l,round(i/5)] = (psd_left[k,l,i-2] +psd_left[k,l,i-1] +psd_left[k,l,i] + psd_left[k,l,i+1] +psd_left[k,l,i+2])/(5)
        return PSD_final,freqs_left

                #avgdata2(i, :, countcond2) = mean(trialspectrum(ind-evaluationsPerBin/2:ind+evaluationsPerBin/2-1,:));
    #print(PSD_final.shape)
    return psd_left,freqs_left
