import numpy as np

class PythonRAUtilities:
    def __init__(self):
        self.t_min = 0.0
        self.index_min = np.int64(0)
        self.temp_ess_ind = np.int64(0)
        self.t_min_ess_index = (0, 0)
        
    def NextState(self, t_min, ng, ness, nl, lambda_tot, mu_tot, current_state, cap_max, cap_min, ess_units):
        self.t_min = t_min
        if self.t_min <= 0:
            self.U = np.random.uniform(0, 1, ng + nl)
            self.time_gt = np.zeros(ng + nl)
            for u in range(ng + nl):
                if current_state[u] == 1:
                    self.time_gt[u] = -np.log(self.U[u])/lambda_tot[u]
                else:
                    self.time_gt[u] = -np.log(self.U[u])/mu_tot[u]

            self.V_fail = np.random.uniform(0, 1, ness)
            self.V_repair = np.random.uniform(0, 1, ness)
            self.time_ess_fail = np.zeros(ness)
            self.time_ess_repair = np.ones(ness)*1e7
            for v in range(ness):
                self.time_ess_fail[v] = -np.log(self.V_fail[v])/lambda_tot[ng + nl + v]
                if current_state[ng + nl + v] < 1:
                    self.time_ess_repair[v] = -np.log(self.V_repair[v])/mu_tot[ng + nl + v]

            self.time_ess = np.vstack((self.time_ess_fail, self.time_ess_repair))
            self.t_min_ess = self.time_ess.min()
            self.t_min_ess_index = np.unravel_index(np.argmin(self.time_ess), self.time_ess.shape)

            self.time_all = np.append(self.time_gt, self.t_min_ess)
            self.t_min = min(self.time_all)
            self.index_min = np.argmin(self.time_all)

            self.temp_ess_ind = self.time_all.size - 1

        self.t_min -= 1

        if self.t_min <= 0 and self.index_min != self.temp_ess_ind:
            if current_state[self.index_min] == 1:
                current_state[self.index_min] = 0
            elif current_state[self.index_min] == 0:
                current_state[self.index_min] = 1
        elif self.t_min <= 0 and self.index_min == self.temp_ess_ind:
            if self.t_min_ess_index[0] == 0:
                self.ess_failed = self.t_min_ess_index[1]
                if current_state[ng + nl + self.ess_failed] >= 1/ess_units[self.ess_failed]:
                    current_state[ng + nl + self.ess_failed] = current_state[ng + nl + self.ess_failed] - 1/ess_units[self.ess_failed]
            else:
                 self.ess_repaired = self.t_min_ess_index[1]
                 if current_state[ng + nl + self.ess_repaired] < 1:
                    current_state[ng + nl + self.ess_repaired] = current_state[ng + nl + self.ess_repaired] + 1/ess_units[self.ess_repaired]

        current_cap = {"max": np.multiply(current_state, cap_max), "min": np.multiply(current_state, cap_min)}
        return(current_state, current_cap, self.t_min)

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
lambda_tot = np.array([0.01, 0.02, 0.015, 0.005, 0.008, 0.012, 0.01], dtype=np.float64) # size ng + nl + ness = 7
mu_tot = np.array([0.1, 0.2, 0.15, 0.05, 0.08, 0.12, 0.1], dtype=np.float64)
cap_max = np.array([100.0, 100.0, 100.0, 50.0, 50.0, 20.0, 20.0], dtype=np.float64)
cap_min = np.array([10.0, 10.0, 10.0, 0.0, 0.0, -10.0, -10.0], dtype=np.float64)
ess_units = np.array([2.0, 2.0], dtype=np.float64)

# Test case 1: Decrement only (t_min > 0)
print("Running Test Case 1 (t_min > 0)...")
current_state_py = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
current_state_rust = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

py_raut = PythonRAUtilities()
# Mock intermediate values on Python side
py_raut.index_min = np.int64(2)
py_raut.temp_ess_ind = 5
py_raut.t_min_ess_index = (0, 0)
py_out_state, py_out_cap, py_out_t = py_raut.NextState(5.5, ng, ness, nl, lambda_tot, mu_tot, current_state_py, cap_max, cap_min, ess_units)

rust_raut = RustRAUtilities()
# Mock intermediate values on Rust side
rust_raut.index_min = 2
rust_raut.temp_ess_ind = 5
rust_raut.t_min_ess_index = (0, 0)
rust_out_state, rust_out_cap, rust_out_t = rust_raut.NextState(5.5, ng, ness, nl, lambda_tot, mu_tot, current_state_rust, cap_max, cap_min, ess_units)

assert py_out_t == rust_out_t == 4.5, f"t_min expected to be 4.5, got Python={py_out_t}, Rust={rust_out_t}"
np.testing.assert_allclose(py_out_state, rust_out_state)
np.testing.assert_allclose(py_out_cap["max"], rust_out_cap["max"])
np.testing.assert_allclose(py_out_cap["min"], rust_out_cap["min"])
print("Test Case 1 Passed!")

# Test case 2: State transition triggered (t_min <= 0)
print("Running Test Case 2 (t_min <= 0)...")
current_state_rust = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
rust_raut = RustRAUtilities()
rust_out_state, rust_out_cap, rust_out_t = rust_raut.NextState(0.0, ng, ness, nl, lambda_tot, mu_tot, current_state_rust, cap_max, cap_min, ess_units)

# Verify types and dimensions
assert isinstance(rust_out_t, float)
assert rust_out_t > 0 # Rust calculated a random time and decremented it by 1, should be > 0 generally unless rates are extremely high
assert isinstance(rust_out_cap, dict)
assert "max" in rust_out_cap and "min" in rust_out_cap
assert len(rust_out_state) == 7
assert len(rust_out_cap["max"]) == 7

# Verify capacity multiplication logic
expected_max = rust_out_state * cap_max
expected_min = rust_out_state * cap_min
np.testing.assert_allclose(rust_out_cap["max"], expected_max)
np.testing.assert_allclose(rust_out_cap["min"], expected_min)

print("Test Case 2 Passed!")
print("All NextState tests passed successfully!")
