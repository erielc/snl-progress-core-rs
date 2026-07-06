# Integrating snl-progress-core-rs into ProGRESS

This guide describes how to deploy and integrate the high-performance Rust-accelerated module (`snl_progress_core_rs`) into the main Python-based ProGRESS (snl-progress) solver framework.

---

## 1. Installation

### From PyPI (End-Users)
Once published to PyPI, end-users do not need Rust or Maturin installed. They can install the pre-compiled wheel directly:
```bash
pip install snl-progress-core-rs
```
To automate this, add `snl-progress-core-rs` to your project's `requirements.txt` or package dependencies.

### From Local Source (Developers)
For local development and testing, install the Rust extension directly into your active Python environment using Maturin:
```bash
maturin develop --release
```
*Note: Do not use standard `cargo build` for final Python integration.*

---

## 2. Drop-in Code Integration

To integrate the Rust library while maintaining a clean fallback to the original Python code (which is useful for development, testing, or unsupported architectures), implement the import using a `try-except` block.

Locate the file in the main `snl-progress` codebase where `RAUtilities` is defined or imported, and update it as follows:

```python
try:
    # 1. Attempt to load the pre-compiled high-performance Rust extension
    from snl_progress_core_rs import RAUtilities
    print("Using Rust-accelerated RAUtilities solver backend.")
except ImportError:
    # 2. Fall back to the original Python-based implementation if the binary is unavailable
    print("Warning: Rust extension not found. Falling back to Python implementation.")
    
    class RAUtilities:
        def __init__(self):
            self.t_min = 0.0
            self.MTTF_all = None
            self.MTTR_all = None
            self.mu_tot = None
            self.lambda_tot = None
            self.cap_min = None
            self.cap_max = None
            self.index_min = 0
            self.temp_ess_ind = 0
            self.t_min_ess_index = (0, 0)
            self.ess_emax = None
            self.ess_smax = None
            self.ess_smin = None
            self.W = None
            self.time_wind = None
            self.w_power = None
            self.w_zones = None
            self.s_zones = None

        def reltrates(self, MTTF_gen, MTTF_trans, MTTR_gen, MTTR_trans, MTTF_ess, MTTR_ess):
            import numpy as np
            self.MTTF_all = np.concatenate((MTTF_gen, MTTF_trans, MTTF_ess))
            self.MTTR_all = np.concatenate((MTTR_gen, MTTR_trans, MTTR_ess))
            self.mu_tot = 1 / self.MTTR_all
            self.lambda_tot = 1 / self.MTTF_all
            return (self.mu_tot, self.lambda_tot)

        # ... (include other original Python methods here as fallbacks)
```

---

## 3. Verifying the Integration

After installation, verify that the Rust implementation is correctly loaded and working by running:

```python
from snl_progress_core_rs import RAUtilities

# Initialize the Rust-backed struct
utilities = RAUtilities()

# Test a method to verify data crosses the FFI boundary correctly
import numpy as np
mttf_gen = np.array([100.0], dtype=np.float64)
mttf_trans = np.array([200.0], dtype=np.float64)
mttf_ess = np.array([50.0], dtype=np.float64)

mttr_gen = np.array([10.0], dtype=np.float64)
mttr_trans = np.array([20.0], dtype=np.float64)
mttr_ess = np.array([5.0], dtype=np.float64)

mu, lam = utilities.reltrates(mttf_gen, mttf_trans, mttr_gen, mttr_trans, mttf_ess, mttr_ess)
print("Rust transition rates computed successfully:", mu, lam)
```
