import unittest

try:
    import numpy as np
    from crest import calculate_pmf_dr
except Exception:  # pragma: no cover
    np = None
    calculate_pmf_dr = None


class TestPMFDR(unittest.TestCase):
    @unittest.skipIf(np is None or calculate_pmf_dr is None, "Optional deps missing (numpy/soundfile)")
    def test_constant_sine_is_about_3db(self):
        sr = 48000
        seconds = 12.0  # 4 blocks of 3s
        t = np.arange(int(sr * seconds)) / sr
        peak = 0.5
        x = peak * np.sin(2 * np.pi * 1000 * t)
        data = x.reshape(-1, 1).astype(np.float32)

        dr = calculate_pmf_dr(data, sr, peak_linear=peak)
        self.assertIsNotNone(dr)
        self.assertEqual(dr["dr_value"], 3)
        self.assertAlmostEqual(dr["dr_db"], 20 * np.log10(np.sqrt(2)), places=2)

    @unittest.skipIf(np is None or calculate_pmf_dr is None, "Optional deps missing (numpy/soundfile)")
    def test_constant_square_is_about_0db(self):
        sr = 44100
        seconds = 9.0  # 3 blocks
        t = np.arange(int(sr * seconds)) / sr
        peak = 0.5
        x = peak * np.sign(np.sin(2 * np.pi * 220 * t))
        data = x.reshape(-1, 1).astype(np.float32)

        dr = calculate_pmf_dr(data, sr, peak_linear=peak)
        self.assertIsNotNone(dr)
        self.assertEqual(dr["dr_value"], 0)
        self.assertAlmostEqual(dr["dr_db"], 0.0, places=2)

    @unittest.skipIf(np is None or calculate_pmf_dr is None, "Optional deps missing (numpy/soundfile)")
    def test_silence_returns_none(self):
        sr = 48000
        data = np.zeros((sr * 6, 2), dtype=np.float32)
        self.assertIsNone(calculate_pmf_dr(data, sr, peak_linear=0.0))
        self.assertIsNone(calculate_pmf_dr(data, sr, peak_linear=1.0))


if __name__ == "__main__":
    unittest.main()
