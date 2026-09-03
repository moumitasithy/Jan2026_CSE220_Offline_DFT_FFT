"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    # TODO: implement this function
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()
    


class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        # TODO: implement this method
        x_arr = np.asarray(x, dtype=np.complex128)
        N = x_arr.shape[0]
        if N == 0:
            return np.array([], dtype=np.complex128)

        n = np.arange(N)
        k = n.reshape((N, 1))
        # Vectorized N x N matrix multiplication for direct evaluation
        twiddles = np.exp(-2j * np.pi * k * n / N)
        return twiddles @ x_arr
        raise NotImplementedError("Implement DFTAnalyzer.transform")

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        # TODO: implement this method
        X = np.asarray(spectrum, dtype=np.complex128)
        N = X.shape[0]
        if N == 0:
            return np.array([], dtype=np.complex128)

        # Standard IDFT property: IDFT(X) = (1/N) * conj(DFT(conj(X)))
        return np.conj(self.transform(np.conj(X))) / N
        raise NotImplementedError("Implement DFTAnalyzer.inverse")


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    # name = "fft"
    # @staticmethod
    # def _is_power_of_two(n):
    #     return n > 0 and (n & (n - 1)) == 0
    # def transform(self, x):
    #     """Forward FFT. Same contract as DFTAnalyzer.transform."""
    #     x_arr = np.asarray(x, dtype=np.complex128).copy()
    #     N = x_arr.shape[0]

    #     if N == 0:
    #         return np.array([], dtype=np.complex128)

    #     if not self._is_power_of_two(N):
    #         raise ValueError(f"Length N={N} must be a power of two for Radix-2 FFT.")

    #     # Bit-reversal permutation
    #     num_bits = N.bit_length() - 1
    #     for i in range(N):
    #         # Reverse bits of i
    #         rev = 0
    #         temp = i
    #         for _ in range(num_bits):
    #             rev = (rev << 1) | (temp & 1)
    #             temp >>= 1
    #         if rev > i:
    #             x_arr[i], x_arr[rev] = x_arr[rev], x_arr[i]

    #     # Iterative Cooley-Tukey Butterfly computation
    #     stage_len = 2
    #     while stage_len <= N:
    #         half_len = stage_len // 2
    #         # Compute stage twiddles ONCE per stage
    #         k = np.arange(half_len)
    #         twiddles = np.exp(-2j * np.pi * k / stage_len)

    #         for i in range(0, N, stage_len):
    #             u = x_arr[i : i + half_len]
    #             v = x_arr[i + half_len : i + stage_len] * twiddles
    #             x_arr[i : i + half_len] = u + v
    #             x_arr[i + half_len : i + stage_len] = u - v

    #         stage_len *= 2

    #     return x_arr
    #     # TODO: implement this method
    #     raise NotImplementedError("Implement FFTTransformer.transform")

    # def inverse(self, spectrum):
    #     """Inverse FFT, including the 1/N factor."""
    #     X = np.asarray(spectrum, dtype=np.complex128)
    #     N = X.shape[0]
    #     if N == 0:
    #         return np.array([], dtype=np.complex128)

    #     return np.conj(self.transform(np.conj(X))) / N
    #     # TODO: implement this method
    #     raise NotImplementedError("Implement FFTTransformer.inverse")
    # class FFTTransformer:
    name = "fft"

    @staticmethod
    def _bit_reverse_indices(n):
        bits = int(np.log2(n))
        indices = np.arange(n)
        reversed_indices = np.zeros(n, dtype=int)
        for i in range(bits):
            reversed_indices = (reversed_indices << 1) | (indices & 1)
            indices >>= 1
        return reversed_indices

    def transform(self, x):
        x = np.asarray(x, dtype=np.complex128)
        N = x.shape[0]
        if N <= 1:
            return x

        # Bit reversal permutation
        rev = self._bit_reverse_indices(N)
        A = x[rev].copy()

        # Iterative Cooley-Tukey
        s = 1
        while (1 << s) <= N:
            m = 1 << s
            m2 = m >> 1
            w_m = np.exp(-2j * np.pi / m)
            
            w_powers = w_m ** np.arange(m2)
            
            for k in range(0, N, m):
                t = w_powers * A[k + m2 : k + m]
                u = A[k : k + m2]
                A[k : k + m2] = u + t
                A[k + m2 : k + m] = u - t
            s += 1

        return A

    def inverse(self, spectrum):
        X = np.asarray(spectrum, dtype=np.complex128)
        N = X.shape[0]
        if N <= 1:
            return X
        return np.conj(self.transform(np.conj(X))) / N

# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        # TODO (bonus): implement this method
        x_arr = np.asarray(x, dtype=np.complex128)
        N = x_arr.shape[0]

        if N == 0:
            return np.array([], dtype=np.complex128)

        # Fast path if N is already a power of 2
        if self._is_power_of_two(N):
            return super().transform(x_arr)

        # Bluestein's chirp-z algorithm
        # N2 length padded to next power of 2 >= 2N - 1
        M = next_power_of_two(2 * N - 1)
        n = np.arange(N)

        # Chirp sequence
        chirp = np.exp(-1j * np.pi * (n ** 2) / N)

        # Form sequence A (length M padded with zeros)
        A = np.zeros(M, dtype=np.complex128)
        A[:N] = x_arr * chirp

        # Form sequence B (length M)
        B = np.zeros(M, dtype=np.complex128)
        B[:N] = np.conj(chirp)
        B[M - N + 1 :] = np.conj(chirp[1:][::-1])

        # Convolution in Frequency Domain using Radix-2 FFT
        fft_engine = FFTTransformer()
        FA = fft_engine.transform(A)
        FB = fft_engine.transform(B)

        # Pointwise product and inverse FFT
        FC = FA * FB
        C = fft_engine.inverse(FC)

        # Truncate and post-multiply by chirp
        return C[:N] * chirp
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.transform")

    def inverse(self, spectrum):
        X = np.asarray(spectrum, dtype=np.complex128)
        N = X.shape[0]
        if N == 0:
            return np.array([], dtype=np.complex128)

        return np.conj(self.transform(np.conj(X))) / N
