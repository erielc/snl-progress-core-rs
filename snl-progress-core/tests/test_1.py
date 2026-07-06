import numpy as np

class PythonRAUtilities:
    def reltrates(self, MTTF_gen, MTTF_trans, MTTR_gen, MTTR_trans, MTTF_ess, MTTR_ess):
        self.MTTF_all = np.concatenate((MTTF_gen, MTTF_trans, MTTF_ess))
        self.MTTR_all = np.concatenate((MTTR_gen, MTTR_trans, MTTR_ess ))
        self.mu_tot = 1/self.MTTR_all
        self.lambda_tot = 1/self.MTTF_all
        return (self.mu_tot, self.lambda_tot)

try:
    from snl_progress_core_rs import RAUtilities as RustRAUtilities # type: ignore
    print("Successfully imported snl_progress_core_rs.RAUtilities!")
except ImportError as e:
    print("Could not import snl_progress_core_rs. Did you run 'maturin develop --release'?")
    raise e

# Create sample inputs
MTTF_gen = np.array([100.0, 150.0], dtype=np.float64)
MTTF_trans = np.array([200.0], dtype=np.float64)
MTTF_ess = np.array([50.0, 75.0], dtype=np.float64)

MTTR_gen = np.array([10.0, 15.0], dtype=np.float64)
MTTR_trans = np.array([20.0], dtype=np.float64)
MTTR_ess = np.array([5.0, 7.5], dtype=np.float64)

# Call Python version
py_raut = PythonRAUtilities()
py_mu, py_lambda = py_raut.reltrates(MTTF_gen, MTTF_trans, MTTR_gen, MTTR_trans, MTTF_ess, MTTR_ess)

# Call Rust version
rust_raut = RustRAUtilities()
rust_mu, rust_lambda = rust_raut.reltrates(MTTF_gen, MTTF_trans, MTTR_gen, MTTR_trans, MTTF_ess, MTTR_ess)

# Assertions
np.testing.assert_allclose(py_mu, rust_mu, err_msg="Repair rates (mu_tot) do not match!")
np.testing.assert_allclose(py_lambda, rust_lambda, err_msg="Failure rates (lambda_tot) do not match!")

# Assert that struct properties are set correctly
np.testing.assert_allclose(py_raut.MTTF_all, rust_raut.MTTF_all, err_msg="MTTF_all properties do not match!")
np.testing.assert_allclose(py_raut.MTTR_all, rust_raut.MTTR_all, err_msg="MTTR_all properties do not match!")
np.testing.assert_allclose(py_raut.mu_tot, rust_raut.mu_tot, err_msg="mu_tot properties do not match!")
np.testing.assert_allclose(py_raut.lambda_tot, rust_raut.lambda_tot, err_msg="lambda_tot properties do not match!")

print("All tests passed! Rust RAUtilities.reltrates matches the Python implementation exactly.")
