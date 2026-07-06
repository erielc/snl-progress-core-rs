import numpy as np

class PythonRAUtilities:
    def updateSOC(self, ng, nl, current_cap, ess_pmax, ess_duration, ess_socmax, ess_socmin, SOC_old):
        self.ess_emax = np.multiply(current_cap["max"][ng + nl::], ess_duration)
        self.ess_smax = np.multiply(self.ess_emax, ess_socmax)
        self.ess_smin = np.multiply(self.ess_emax, ess_socmin)
        SOC_old = current_cap["max"][ng + nl::]*SOC_old/ess_pmax
        return (self.ess_smax, self.ess_smin, SOC_old)

try:
    from snl_progress_core_rs import RAUtilities as RustRAUtilities
    print("Successfully imported snl_progress_core_rs.RAUtilities!")
except ImportError as e:
    print("Could not import snl_progress_core_rs. Did you run 'maturin develop --release'?")
    raise e

# Setup parameters
ng = 3
nl = 2
ness = 2
current_cap = {
    "max": np.array([100.0, 100.0, 100.0, 50.0, 50.0, 20.0, 20.0], dtype=np.float64) # size 7, ng+nl = 5, ESS starts at index 5 (size 2)
}
ess_pmax = np.array([20.0, 10.0], dtype=np.float64)
ess_duration = np.array([4.0, 6.0], dtype=np.float64)
ess_socmax = np.array([0.9, 0.95], dtype=np.float64)
ess_socmin = np.array([0.1, 0.15], dtype=np.float64)
SOC_old = np.array([15.0, 8.0], dtype=np.float64)

# Call Python version
py_raut = PythonRAUtilities()
py_smax, py_smin, py_soc = py_raut.updateSOC(ng, nl, current_cap, ess_pmax, ess_duration, ess_socmax, ess_socmin, SOC_old)

# Call Rust version
rust_raut = RustRAUtilities()
rust_smax, rust_smin, rust_soc = rust_raut.updateSOC(ng, nl, current_cap, ess_pmax, ess_duration, ess_socmax, ess_socmin, SOC_old)

# Assertions
np.testing.assert_allclose(py_smax, rust_smax, err_msg="ess_smax do not match!")
np.testing.assert_allclose(py_smin, rust_smin, err_msg="ess_smin do not match!")
np.testing.assert_allclose(py_soc, rust_soc, err_msg="SOC_old do not match!")

# Assert that struct properties are set correctly
np.testing.assert_allclose(py_raut.ess_emax, rust_raut.ess_emax, err_msg="ess_emax properties do not match!")
np.testing.assert_allclose(py_raut.ess_smax, rust_raut.ess_smax, err_msg="ess_smax properties do not match!")
np.testing.assert_allclose(py_raut.ess_smin, rust_raut.ess_smin, err_msg="ess_smin properties do not match!")

print("All tests passed! Rust RAUtilities.updateSOC matches the Python implementation exactly.")
