#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyDict, PyList};
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray1,
    PyReadwriteArray3, PyReadwriteArrayDyn, ToPyArray, PyArray1, PyArray2,
    PyArrayMethods,
};
use rand::Rng;

// Python<'py>  is a special PyO3 token (or proof marker) that represents ownership of the Python Global Interpreter Lock (GIL).

/// This class contains the different methods required for performing mixed time sequential Monte Carlo simulation and evaluate the reliability indices of a power system.
#[pyclass(subclass)]
#[derive(Default)]
pub struct RAUtilities {
    /// The minimum time remaining or elapsed for the current simulation state.
    #[pyo3(get, set)]
    pub t_min: f64,

    /// Mean Time to Failure (MTTF) concatenated for all components.
    #[pyo3(get, set)]
    pub MTTF_all: Option<Py<PyAny>>,

    /// Mean Time to Repair (MTTR) concatenated for all components.
    #[pyo3(get, set)]
    pub MTTR_all: Option<Py<PyAny>>,

    /// Total repair rates for all components (1 / MTTR_all).
    #[pyo3(get, set)]
    pub mu_tot: Option<Py<PyAny>>,

    /// Total failure rates for all components (1 / MTTF_all).
    #[pyo3(get, set)]
    pub lambda_tot: Option<Py<PyAny>>,

    /// Minimum capacities concatenated for all components.
    #[pyo3(get, set)]
    pub cap_min: Option<Py<PyAny>>,

    /// Maximum capacities concatenated for all components.
    #[pyo3(get, set)]
    pub cap_max: Option<Py<PyAny>>,

    /// Internal state for component index with the minimum time to next failure/repair.
    #[pyo3(get, set)]
    pub index_min: usize,

    /// Internal state for index of ESS component group.
    #[pyo3(get, set)]
    pub temp_ess_ind: usize,

    /// Internal state for row and column of the next ESS component failure/repair event.
    #[pyo3(get, set)]
    pub t_min_ess_index: (usize, usize),

    /// Maximum energy capacity for ESS components.
    #[pyo3(get, set)]
    pub ess_emax: Option<Py<PyAny>>,

    /// Maximum allowable State-of-Charge (SOC) as energy.
    #[pyo3(get, set)]
    pub ess_smax: Option<Py<PyAny>>,

    /// Minimum allowable State-of-Charge (SOC) as energy.
    #[pyo3(get, set)]
    pub ess_smin: Option<Py<PyAny>>,

    /// Uniform random numbers generated for each wind class at each site.
    #[pyo3(get, set)]
    pub W: Option<Py<PyAny>>,

    /// Minimum time to next state transition for each wind speed class at each site.
    #[pyo3(get, set)]
    pub time_wind: Option<Py<PyAny>>,

    /// Wind power generation output for each site.
    #[pyo3(get, set)]
    pub w_power: Option<Py<PyAny>>,

    /// Aggregated wind power generation by grid zone.
    #[pyo3(get, set)]
    pub w_zones: Option<Py<PyAny>>,

    /// Cached solar power generation profile by zone and hour.
    #[pyo3(get, set)]
    pub s_zones: Option<Py<PyAny>>,
}

#[pymethods]
impl RAUtilities {
    /// Initializes the RAUtilities class.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculates failure and repair rates for all conventional generators and transmission lines.
    ///
    /// Parameters:
    ///     MTTF_gen (PyReadonlyArray1<f64>): Mean time to failure for generators.
    ///     MTTF_trans (PyReadonlyArray1<f64>): Mean time to failure for transmission lines.
    ///     MTTR_gen (PyReadonlyArray1<f64>): Mean time to repair for generators.
    ///     MTTR_trans (PyReadonlyArray1<f64>): Mean time to repair for transmission lines.
    ///     MTTF_ess (PyReadonlyArray1<f64>): Mean time to failure for energy storage systems.
    ///     MTTR_ess (PyReadonlyArray1<f64>): Mean time to repair for energy storage systems.
    ///
    /// Returns:
    ///     (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>): Repair rates and failure rates for all components.
    #[pyo3(signature = (MTTF_gen, MTTF_trans, MTTR_gen, MTTR_trans, MTTF_ess, MTTR_ess))]
    pub fn reltrates<'py>(
        &mut self,
        py: Python<'py>,
        MTTF_gen: PyReadonlyArray1<'py, f64>,
        MTTF_trans: PyReadonlyArray1<'py, f64>,
        MTTR_gen: PyReadonlyArray1<'py, f64>,
        MTTR_trans: PyReadonlyArray1<'py, f64>,
        MTTF_ess: PyReadonlyArray1<'py, f64>,
        MTTR_ess: PyReadonlyArray1<'py, f64>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        // Convert numpy array arguments to contiguous Rust slices
        let mttf_gen = MTTF_gen.as_slice().unwrap();
        let mttf_trans = MTTF_trans.as_slice().unwrap();
        let mttf_ess = MTTF_ess.as_slice().unwrap();

        let mttr_gen = MTTR_gen.as_slice().unwrap();
        let mttr_trans = MTTR_trans.as_slice().unwrap();
        let mttr_ess = MTTR_ess.as_slice().unwrap();

        // Concatenate MTTF values using Rust iterators to avoid multiple intermediate allocations
        let mttf_all: Vec<f64> = mttf_gen.iter().copied()
            .chain(mttf_trans.iter().copied())
            .chain(mttf_ess.iter().copied())
            .collect();

        // Concatenate MTTR values
        let mttr_all: Vec<f64> = mttr_gen.iter().copied()
            .chain(mttr_trans.iter().copied())
            .chain(mttr_ess.iter().copied())
            .collect();

        // Compute reciprocals (repair rates and failure rates)
        let mu_tot: Vec<f64> = mttr_all.iter().copied().map(|x| 1.0 / x).collect();
        let lambda_tot: Vec<f64> = mttf_all.iter().copied().map(|x| 1.0 / x).collect();

        // Convert the Rust vectors directly to NumPy arrays on the Python heap
        let mttf_all_py = mttf_all.to_pyarray(py);
        let mttr_all_py = mttr_all.to_pyarray(py);
        let lambda_tot_py = lambda_tot.to_pyarray(py);
        let mu_tot_py = mu_tot.to_pyarray(py);

        // Store these arrays inside the struct fields (converting Bound references to GIL-independent PyObjects)
        self.MTTF_all = Some(mttf_all_py.clone().into_any().unbind());
        self.MTTR_all = Some(mttr_all_py.clone().into_any().unbind());
        self.lambda_tot = Some(lambda_tot_py.clone().into_any().unbind());
        self.mu_tot = Some(mu_tot_py.clone().into_any().unbind());

        // Return the repair and failure rates as a tuple
        (mu_tot_py, lambda_tot_py)
    }

    /// Concatenates capacities of generators and transmission lines for use in the MCS.
    ///
    /// Parameters:
    ///     nl (usize): Number of lines.
    ///     pmax (PyReadonlyArray1<f64>): Maximum capacities of generators.
    ///     pmin (PyReadonlyArray1<f64>): Minimum capacities of generators.
    ///     ess_pmax (PyReadonlyArray1<f64>): Maximum capacities of energy storage systems.
    ///     ess_pmin (PyReadonlyArray1<f64>): Minimum capacities of energy storage systems.
    ///     cap_trans (PyReadonlyArray1<f64>): Transmission capacities.
    ///
    /// Returns:
    ///     (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>): Maximum and minimum capacities of all components.
    #[pyo3(signature = (nl, pmax, pmin, ess_pmax, ess_pmin, cap_trans))]
    pub fn capacities<'py>(
        &mut self,
        py: Python<'py>,
        nl: usize,
        pmax: PyReadonlyArray1<'py, f64>,
        pmin: PyReadonlyArray1<'py, f64>,
        ess_pmax: PyReadonlyArray1<'py, f64>,
        ess_pmin: PyReadonlyArray1<'py, f64>,
        cap_trans: PyReadonlyArray1<'py, f64>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        // Convert numpy array arguments to contiguous Rust slices
        let pmax_slice = pmax.as_slice().unwrap();
        let pmin_slice = pmin.as_slice().unwrap();
        let ess_pmax_slice = ess_pmax.as_slice().unwrap();
        let ess_pmin_slice = ess_pmin.as_slice().unwrap();
        let cap_trans_slice = cap_trans.as_slice().unwrap();

        // Concatenate cap_max = (pmax, cap_trans, ess_pmax)
        let cap_max: Vec<f64> = pmax_slice.iter().copied()
            .chain(cap_trans_slice.iter().copied())
            .chain(ess_pmax_slice.iter().copied())
            .collect();

        // Concatenate cap_min = (pmin, np.zeros(nl), ess_pmin)
        // std::iter::repeat(0.0).take(nl) generates `nl` zeroes on-the-fly without allocating memory.
        let cap_min: Vec<f64> = pmin_slice.iter().copied()
            .chain(std::iter::repeat(0.0).take(nl))
            .chain(ess_pmin_slice.iter().copied())
            .collect();

        // Convert the Rust vectors directly to NumPy arrays on the Python heap
        let cap_max_py = cap_max.to_pyarray(py);
        let cap_min_py = cap_min.to_pyarray(py);

        // Store these arrays inside the struct fields
        self.cap_max = Some(cap_max_py.clone().into_any().unbind());
        self.cap_min = Some(cap_min_py.clone().into_any().unbind());

        // Return the capacity arrays as a tuple
        (cap_max_py, cap_min_py)
    }

    /// Generates random numbers to calculate the time to the next state for generators and transmission lines.
    ///
    /// Parameters:
    ///     t_min (f64): Minimum time.
    ///     ng (usize): Number of generators.
    ///     ness (usize): Number of energy storage systems.
    ///     nl (usize): Number of lines.
    ///     lambda_tot (PyReadonlyArray1<f64>): Failure rates.
    ///     mu_tot (PyReadonlyArray1<f64>): Repair rates.
    ///     current_state (PyReadwriteArray1<f64>): Current states of components.
    ///     cap_max (PyReadonlyArray1<f64>): Maximum capacities of components.
    ///     cap_min (PyReadonlyArray1<f64>): Minimum capacities of components.
    ///     ess_units (PyReadonlyArray1<f64>): Units of energy storage systems.
    ///
    /// Returns:
    ///     (Bound<'py, PyAny>, Bound<'py, PyDict>, f64): Current state, current capacity dict, and minimum time.
    #[pyo3(signature = (t_min, ng, ness, nl, lambda_tot, mu_tot, current_state, cap_max, cap_min, ess_units))]
    pub fn NextState<'py>(
        &mut self,
        py: Python<'py>,
        t_min: f64,
        ng: usize,
        ness: usize,
        nl: usize,
        lambda_tot: PyReadonlyArray1<'py, f64>,
        mu_tot: PyReadonlyArray1<'py, f64>,
        mut current_state: PyReadwriteArray1<'py, f64>,
        cap_max: PyReadonlyArray1<'py, f64>,
        cap_min: PyReadonlyArray1<'py, f64>,
        ess_units: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>, f64)> {
        let lambda_slice = lambda_tot.as_slice().unwrap();
        let mu_slice = mu_tot.as_slice().unwrap();
        let mut current_state_view = current_state.as_array_mut();
        let ess_units_slice = ess_units.as_slice().unwrap();
        let cap_max_slice = cap_max.as_slice().unwrap();
        let cap_min_slice = cap_min.as_slice().unwrap();

        self.t_min = t_min;

        if self.t_min <= 0.0 {
            let mut rng = rand::thread_rng();

            // 1. Calculate time to next failure/repair for generators and transmission lines (gt)
            let mut time_gt = vec![0.0; ng + nl];
            for u in 0..(ng + nl) {
                let u_val: f64 = rng.gen_range(0.0..1.0);
                if current_state_view[u] == 1.0 {
                    time_gt[u] = -u_val.ln() / lambda_slice[u];
                } else {
                    time_gt[u] = -u_val.ln() / mu_slice[u];
                }
            }

            // 2. Calculate time for ESS states failure/repair
            let mut time_ess_fail = vec![0.0; ness];
            let mut time_ess_repair = vec![1e7; ness];
            for v in 0..ness {
                let v_fail: f64 = rng.gen_range(0.0..1.0);
                let v_repair: f64 = rng.gen_range(0.0..1.0);
                
                time_ess_fail[v] = -v_fail.ln() / lambda_slice[ng + nl + v];
                if current_state_view[ng + nl + v] < 1.0 {
                    time_ess_repair[v] = -v_repair.ln() / mu_slice[ng + nl + v];
                }
            }

            // Find minimum transition time for ESS components and its row/col index
            let mut t_min_ess = f64::INFINITY;
            let mut t_min_ess_row = 0;
            let mut t_min_ess_col = 0;
            for v in 0..ness {
                if time_ess_fail[v] < t_min_ess {
                    t_min_ess = time_ess_fail[v];
                    t_min_ess_row = 0;
                    t_min_ess_col = v;
                }
                if time_ess_repair[v] < t_min_ess {
                    t_min_ess = time_ess_repair[v];
                    t_min_ess_row = 1;
                    t_min_ess_col = v;
                }
            }
            self.t_min_ess_index = (t_min_ess_row, t_min_ess_col);

            // Find the shortest transition time overall (generators + transmission + ESS)
            let mut index_min = 0;
            let mut min_val = f64::INFINITY;
            for u in 0..(ng + nl) {
                if time_gt[u] < min_val {
                    min_val = time_gt[u];
                    index_min = u;
                }
            }

            let temp_ess_ind = ng + nl;
            if t_min_ess < min_val {
                min_val = t_min_ess;
                index_min = temp_ess_ind;
            }

            self.t_min = min_val;
            self.index_min = index_min;
            self.temp_ess_ind = temp_ess_ind;
        }

        self.t_min -= 1.0;

        // If time reaches <= 0, trigger the transition
        if self.t_min <= 0.0 {
            if self.index_min != self.temp_ess_ind {
                let idx = self.index_min;
                if current_state_view[idx] == 1.0 {
                    current_state_view[idx] = 0.0;
                } else if current_state_view[idx] == 0.0 {
                    current_state_view[idx] = 1.0;
                }
            } else {
                let (row, col) = self.t_min_ess_index;
                let idx = ng + nl + col;
                let unit_val = 1.0 / ess_units_slice[col];
                if row == 0 {
                    if current_state_view[idx] >= unit_val {
                        current_state_view[idx] -= unit_val;
                    }
                } else {
                    if current_state_view[idx] < 1.0 {
                        current_state_view[idx] += unit_val;
                    }
                }
            }
        }

        // Calculate current capacities of all components (current_state * cap)
        let state_len = current_state_view.len();
        let mut cap_max_mult = Vec::with_capacity(state_len);
        let mut cap_min_mult = Vec::with_capacity(state_len);
        for i in 0..state_len {
            cap_max_mult.push(current_state_view[i] * cap_max_slice[i]);
            cap_min_mult.push(current_state_view[i] * cap_min_slice[i]);
        }

        let cap_max_mult_py = cap_max_mult.to_pyarray(py);
        let cap_min_mult_py = cap_min_mult.to_pyarray(py);

        let current_cap_dict = PyDict::new(py);
        current_cap_dict.set_item("max", cap_max_mult_py)?;
        current_cap_dict.set_item("min", cap_min_mult_py)?;

        let current_state_any = current_state.as_any().clone();

        Ok((current_state_any, current_cap_dict, self.t_min))
    }

    /// Updates the allowable state of charge (SOC) limits and rescales the current SOC.
    ///
    /// Parameters:
    ///     ng (usize): Number of generators.
    ///     nl (usize): Number of lines.
    ///     current_cap (Bound<'py, PyDict>): Current capacities of components.
    ///     ess_pmax (PyReadonlyArray1<f64>): Maximum power outputs of energy storage systems.
    ///     ess_duration (PyReadonlyArray1<f64>): Durations of energy storage systems.
    ///     ess_socmax (PyReadonlyArray1<f64>): Maximum state of charge of energy storage systems.
    ///     ess_socmin (PyReadonlyArray1<f64>): Minimum state of charge of energy storage systems.
    ///     SOC_old (PyReadonlyArray1<f64>): Previous state of charge.
    ///
    /// Returns:
    ///     (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>): Max SOC, Min SOC, and updated SOC.
    #[pyo3(signature = (ng, nl, current_cap, ess_pmax, ess_duration, ess_socmax, ess_socmin, SOC_old))]
    pub fn updateSOC<'py>(
        &mut self,
        py: Python<'py>,
        ng: usize,
        nl: usize,
        current_cap: Bound<'py, PyDict>,
        ess_pmax: PyReadonlyArray1<'py, f64>,
        ess_duration: PyReadonlyArray1<'py, f64>,
        ess_socmax: PyReadonlyArray1<'py, f64>,
        ess_socmin: PyReadonlyArray1<'py, f64>,
        SOC_old: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        // Extract "max" array from the current_cap dictionary
        let max_val = current_cap.get_item("max")?.unwrap();
        let max_array: PyReadonlyArray1<'py, f64> = max_val.extract()?;
        let max_slice = max_array.as_slice().unwrap();

        // Slice from ng + nl onwards (which represents the ESS component capacities)
        let current_cap_ess = &max_slice[(ng + nl)..];

        let ess_pmax_slice = ess_pmax.as_slice().unwrap();
        let ess_duration_slice = ess_duration.as_slice().unwrap();
        let ess_socmax_slice = ess_socmax.as_slice().unwrap();
        let ess_socmin_slice = ess_socmin.as_slice().unwrap();
        let soc_old_slice = SOC_old.as_slice().unwrap();

        let ness = current_cap_ess.len();

        let mut ess_emax = Vec::with_capacity(ness);
        let mut ess_smax = Vec::with_capacity(ness);
        let mut ess_smin = Vec::with_capacity(ness);
        let mut updated_soc = Vec::with_capacity(ness);

        for i in 0..ness {
            let emax = current_cap_ess[i] * ess_duration_slice[i];
            ess_emax.push(emax);
            ess_smax.push(emax * ess_socmax_slice[i]);
            ess_smin.push(emax * ess_socmin_slice[i]);
            updated_soc.push(current_cap_ess[i] * soc_old_slice[i] / ess_pmax_slice[i]);
        }

        let ess_smax_py = ess_smax.to_pyarray(py);
        let ess_smin_py = ess_smin.to_pyarray(py);
        let updated_soc_py = updated_soc.to_pyarray(py);

        // Store intermediate fields in the struct
        self.ess_emax = Some(ess_emax.to_pyarray(py).into_any().unbind());
        self.ess_smax = Some(ess_smax_py.clone().into_any().unbind());
        self.ess_smin = Some(ess_smin_py.clone().into_any().unbind());

        Ok((ess_smax_py, ess_smin_py, updated_soc_py))
    }

    /// Calculates the wind power generation for each hour at each site.
    ///
    /// Parameters:
    ///     nz (usize): Number of zones.
    ///     w_sites (usize): Number of wind sites.
    ///     zone_no (PyReadonlyArray1<i64>): Zone numbers.
    ///     w_classes (usize): Number of wind speed classes.
    ///     r_cap (PyReadonlyArray1<f64>): Rated capacities of wind turbines.
    ///     current_w_class (PyReadwriteArray1<i64>): Current wind speed classes.
    ///     tr_mats (PyReadwriteArray3<f64>): Transition matrices.
    ///     p_class (PyReadonlyArray1<i64>): Power classes.
    ///     w_turbines (PyReadonlyArray1<f64>): Number of wind turbines.
    ///     out_curve2 (PyReadonlyArray1<f64>): Output curve for class 2 turbines.
    ///     out_curve3 (PyReadonlyArray1<f64>): Output curve for class 3 turbines.
    ///
    /// Returns:
    ///     (Bound<'py, PyArray1<f64>>, Bound<'py, PyAny>): Wind power generation by zone and updated wind classes.
    #[pyo3(signature = (nz, w_sites, zone_no, w_classes, r_cap, current_w_class, tr_mats, p_class, w_turbines, out_curve2, out_curve3))]
    pub fn WindPower<'py>(
        &mut self,
        py: Python<'py>,
        nz: usize,
        w_sites: usize,
        zone_no: PyReadonlyArray1<'py, i64>,
        w_classes: usize,
        r_cap: PyReadonlyArray1<'py, f64>,
        mut current_w_class: PyReadwriteArrayDyn<'py, i64>,
        mut tr_mats: PyReadwriteArray3<'py, f64>,
        p_class: PyReadonlyArray1<'py, i64>,
        w_turbines: PyReadonlyArray1<'py, f64>,
        out_curve2: PyReadonlyArray1<'py, f64>,
        out_curve3: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyAny>)> {
        let mut rng = rand::thread_rng();

        // 1. Generate uniform random numbers for each class in each site
        let mut W_vec: Vec<f64> = Vec::with_capacity(w_sites * w_classes);
        for _ in 0..(w_sites * w_classes) {
            W_vec.push(rng.gen_range(0.0..1.0));
        }
        let W_py = PyArray1::from_vec(py, W_vec.clone())
            .reshape([w_sites, w_classes])
            .unwrap();

        // 2. Modify transition matrices in-place: change zeroes to 1e-10
        let mut tr_mats_view = tr_mats.as_array_mut();
        tr_mats_view.mapv_inplace(|x| if x == 0.0 { 1e-10 } else { x });

        // 3. Calculate minimum time to next state for each site
        let current_w_class_view = current_w_class.as_array();
        let ndim = current_w_class_view.ndim();

        // Helper to get class dynamically based on dimensions (1D or 2D)
        let get_w_class = |w: usize| -> usize {
            if ndim == 1 {
                current_w_class_view[[w].as_slice()] as usize
            } else {
                current_w_class_view[[w, 0].as_slice()] as usize
            }
        };

        let mut time_wind = vec![0.0; w_sites * w_classes];
        for w in 0..w_sites {
            let c_active = get_w_class(w);
            for c_next in 0..w_classes {
                let w_idx = w * w_classes + c_next;
                let tr_val = tr_mats_view[[w, c_active, c_next]];
                time_wind[w_idx] = -W_vec[w_idx].ln() / tr_val;
            }
        }

        // Find the index of the minimum time for each site (argmin along axis 1)
        let mut tmin_wind = vec![0i64; w_sites];
        for w in 0..w_sites {
            let mut min_val = f64::INFINITY;
            let mut min_c = 0;
            for c in 0..w_classes {
                let val = time_wind[w * w_classes + c];
                if val < min_val {
                    min_val = val;
                    min_c = c;
                }
            }
            tmin_wind[w] = min_c as i64;
        }

        // Mutate current_w_class in-place
        let mut current_w_class_mut = current_w_class.as_array_mut();
        for w in 0..w_sites {
            if ndim == 1 {
                current_w_class_mut[[w].as_slice()] = tmin_wind[w];
            } else {
                current_w_class_mut[[w, 0].as_slice()] = tmin_wind[w];
            }
        }

        // 4. Calculate wind power generation at each site
        let p_class_slice = p_class.as_slice().unwrap();
        let w_turbines_slice = w_turbines.as_slice().unwrap();
        let r_cap_slice = r_cap.as_slice().unwrap();
        let out_curve2_slice = out_curve2.as_slice().unwrap();
        let out_curve3_slice = out_curve3.as_slice().unwrap();

        let mut w_power = vec![0.0; w_sites];
        for w in 0..w_sites {
            let c_min = tmin_wind[w] as usize;
            if p_class_slice[w] == 2 {
                w_power[w] = out_curve2_slice[c_min] * w_turbines_slice[w] * r_cap_slice[w];
            } else {
                w_power[w] = out_curve3_slice[c_min] * w_turbines_slice[w] * r_cap_slice[w];
            }
        }

        // 5. Aggregate wind power generation to grid zones
        let zone_no_slice = zone_no.as_slice().unwrap();
        let mut w_zones = vec![0.0; nz];
        for z in 0..zone_no_slice.len() {
            let zone_idx = (zone_no_slice[z] - 1) as usize;
            if zone_idx < nz {
                w_zones[zone_idx] += w_power[z];
            }
        }

        let w_zones_py = w_zones.to_pyarray(py);

        // Save intermediate fields to struct (reshape time_wind to shape [w_sites, w_classes])
        let time_wind_py = time_wind.to_pyarray(py)
            .reshape([w_sites, w_classes]).unwrap();
        self.time_wind = Some(time_wind_py.clone().into_any().unbind());
        self.W = Some(W_py.clone().into_any().unbind());
        self.w_power = Some(w_power.to_pyarray(py).into_any().unbind());
        self.w_zones = Some(w_zones_py.clone().into_any().unbind());

        let current_w_class_any = current_w_class.as_any().clone();

        Ok((w_zones_py, current_w_class_any))
    }

    /// Calculates solar power generation for each hour at each site.
    ///
    /// Parameters:
    ///     n (usize): Current hour.
    ///     nz (usize): Number of zones.
    ///     s_zone_no (PyReadonlyArray1<i64>): Zone numbers for solar sites.
    ///     solar_prob (PyReadonlyArray2<f64>): Solar probability data.
    ///     s_profiles (Bound<'py, PyList>): Solar profiles.
    ///     s_sites (usize): Number of solar sites.
    ///     s_max (PyReadonlyArray1<f64>): Maximum capacities of solar sites.
    ///
    /// Returns:
    ///     Bound<'py, PyAny>: Transposed solar power generation at each zone.
    #[pyo3(signature = (n, nz, s_zone_no, solar_prob, s_profiles, s_sites, s_max))]
    pub fn SolarPower<'py>(
        &mut self,
        py: Python<'py>,
        n: usize,
        nz: usize,
        s_zone_no: PyReadonlyArray1<'py, i64>,
        solar_prob: PyReadonlyArray2<'py, f64>,
        s_profiles: Bound<'py, PyList>,
        s_sites: usize,
        s_max: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = rand::thread_rng();

        if n % 24 == 0 {
            let month = ((n / 731) as usize) % 12;
            let num_clusters = solar_prob.as_array().shape()[0];
            let solar_prob_view = solar_prob.as_array();

            // Pair each probability with its index
            let mut prob_index: Vec<(f64, usize)> = (0..num_clusters)
                .map(|i| (solar_prob_view[[i, month]], i))
                .collect();

            // Sort by probability ascending
            prob_index.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Cumulative sum
            let mut cum_sum = 0.0;
            let mut sorted_prob = Vec::with_capacity(num_clusters);
            for &(prob, idx) in &prob_index {
                cum_sum += prob;
                sorted_prob.push((cum_sum, idx));
            }

            // Sample random cluster
            let rand_clust: f64 = rng.gen_range(0.0..1.0);
            let mut clust = 0;
            for i in 0..num_clusters {
                if i == 0 {
                    if rand_clust < sorted_prob[i].0 {
                        clust = sorted_prob[i].1;
                        break;
                    }
                } else {
                    if sorted_prob[i - 1].0 <= rand_clust && rand_clust < sorted_prob[i].0 {
                        clust = sorted_prob[i].1;
                        break;
                    }
                }
            }

            // Extract the 3D array: s_profiles[clust]
            let cluster_arr_any = s_profiles.get_item(clust)?;
            let cluster_arr: PyReadonlyArray3<'py, f64> = cluster_arr_any.extract()?;
            let cluster_view = cluster_arr.as_array(); // shape (days, 24, s_sites)

            let days = cluster_view.shape()[0];
            let rand_day: usize = rng.gen_range(0..days);

            // Compute sgen_sites of shape (s_sites, 24)
            let s_max_slice = s_max.as_slice().unwrap();
            let mut sgen_sites = vec![0.0; s_sites * 24];
            for sg in 0..s_sites {
                let max_cap = s_max_slice[sg];
                for hour in 0..24 {
                    let val = cluster_view[[rand_day, hour, sg]];
                    sgen_sites[sg * 24 + hour] = val * max_cap;
                }
            }

            // Aggregate solar generation by zone (s_zones of shape nz x 24)
            let s_zone_no_slice = s_zone_no.as_slice().unwrap();
            let mut s_zones = vec![0.0; nz * 24];
            for z in 0..s_zone_no_slice.len() {
                let zone_idx = (s_zone_no_slice[z] - 1) as usize;
                if zone_idx < nz {
                    for hour in 0..24 {
                        s_zones[zone_idx * 24 + hour] += sgen_sites[z * 24 + hour];
                    }
                }
            }

            let s_zones_py = s_zones.to_pyarray(py).reshape([nz, 24]).unwrap();
            self.s_zones = Some(s_zones_py.clone().into_any().unbind());
        }

        // Return the transpose of self.s_zones
        let s_zones_py = self.s_zones.as_ref().unwrap().bind(py);
        let s_zones_arr = s_zones_py.cast::<PyArray2<f64>>()?;
        let transposed = s_zones_arr.transpose().unwrap();

        Ok(transposed.into_any())
    }
      
}

/// The snl_progress_core_rs Python extension module implemented in Rust.
#[pymodule]
fn snl_progress_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RAUtilities>()?;
    Ok(())
}