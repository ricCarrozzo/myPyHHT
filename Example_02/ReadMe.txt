
#
# Frequency modulated signal + Random Noise + time-varying secular term
#

Quite complex example which demonstrates the powerful filtering capabilities of the Empirical Mode Decomposition process.

The 3 data files are snapshots of the 1st set of Hilbert-Huang iterations, with sifting convergence limits set to 1e-8, 1e-6 and 1e-4 respectively (using the last signal array of the previous set as starting point). 

After the 33rd iteration, when the residual FFT spectrum is essentially bare of any frequency above 1Hz, the residual signal recovers pretty much the whole secular term.
Subtracting it to the original signal allows a further round of iterations to determine the frequency components.

The removal of the low frequency bias exposes the 2Hz frequency mode, which can be filtered out easily.
Then a set of iterations with stringent sifting convergence limits will remove most of the chuff due to the random noise.

The process here requires a bit of tinkering, but looking at the reconstructed phase, unwrapped from the one estimated via Hilbert Transform steps, it's quite clear how even such a tricky signal can be treated effectively.

The instantaneous frequency estimate confirms a linear trend, while fitting a 3rd degree polynomial to the unwrapped phase will return a decent approximation of the orginal input one.

Considering the relatively low sampling frequency (1KHz), that looks quite impressive, despite the naive implementation given.
   
