import numpy as np
import unittest

from fft import fft, ifft


class FFTTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.random([2, 8, 4, 2048])

    def testFFT1D(self):
        expected = np.fft.fft(self.x)
        actual = fft(self.x, -1)
        self.assertTrue(np.allclose(expected, actual))

        expected = np.fft.ifft(expected)
        actual = ifft(actual, -1)
        self.assertTrue(np.allclose(expected, actual))

    def testFFT2D(self):
        expected = np.fft.fft2(self.x)
        actual = fft(self.x, -2)
        self.assertTrue(np.allclose(expected, actual))

        expected = np.fft.ifft(expected)
        actual = ifft(actual, -2)
        self.assertTrue(np.allclose(expected, actual))

    def testFFT(self):
        expected = np.fft.fftn(self.x)
        actual = fft(self.x)
        self.assertTrue(np.allclose(expected, actual))

        expected = np.fft.ifftn(expected)
        actual = ifft(actual)
        self.assertTrue(np.allclose(expected, actual))


if __name__ == "__main__":
    unittest.main()
