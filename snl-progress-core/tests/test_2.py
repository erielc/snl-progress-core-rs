import numpy as np

class PythonRAUtilities:
    def capacities(self, nl, pmax, pmin, ess_pmax, ess_pmin, cap_trans):
        self.cap_max = np.concatenate((pmax, cap_trans, ess_pmax))
        self.cap_min = np.concatenate((pmin, np.zeros(nl), ess_pmin))
        return (self.cap_max, self.cap_min)

try:
    from snl_progress_core_rs import RAUtilities as RustRAUtilities # type:ignore
    print("Successfully imported snl_progress_core_rs.RAUtilities!")
except ImportError as e:
    print("Could not import snl_progress_core_rs. Did you run 'maturin develop --release'?")
    raise e

# Create sample inputs
nl = 5
pmax = np.array([100.0, 150.0], dtype=np.float64)
pmin = np.array([10.0, 15.0], dtype=np.float64)
ess_pmax = np.array([50.0, 75.0], dtype=np.float64)
ess_pmin = np.array([-50.0, -75.0], dtype=np.float64)
cap_trans = np.array([200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float64)

# Call Python version
py_raut = PythonRAUtilities()
py_max, py_min = py_raut.capacities(nl, pmax, pmin, ess_pmax, ess_pmin, cap_trans)

# Call Rust version
rust_raut = RustRAUtilities()
rust_max, rust_min = rust_raut.capacities(nl, pmax, pmin, ess_pmax, ess_pmin, cap_trans)

# Assertions
np.testing.assert_allclose(py_max, rust_max, err_msg="Max capacities (cap_max) do not match!")
np.testing.assert_allclose(py_min, rust_min, err_msg="Min capacities (cap_min) do not match!")

# Assert that struct properties are set correctly
np.testing.assert_allclose(py_raut.cap_max, rust_raut.cap_max, err_msg="cap_max properties do not match!")
np.testing.assert_allclose(py_raut.cap_min, rust_raut.cap_min, err_msg="cap_min properties do not match!")

print("All tests passed! Rust RAUtilities.capacities matches the Python implementation exactly.")
