import time
import numpy as np

# Pure Python copies of the methods
class PythonRAUtilities:
    def __init__(self):
        self.t_min = 0.0
        self.s_zones = None

    def reltrates(self, MTTF_gen, MTTF_trans, MTTR_gen, MTTR_trans, MTTF_ess, MTTR_ess):
        self.MTTF_all = np.concatenate((MTTF_gen, MTTF_trans, MTTF_ess))
        self.MTTR_all = np.concatenate((MTTR_gen, MTTF_trans, MTTR_ess)) # note: match capacities shape
        self.mu_tot = 1/self.MTTR_all
        self.lambda_tot = 1/self.MTTF_all
        return(self.mu_tot, self.lambda_tot)

    def capacities(self, nl, pmax, pmin, ess_pmax, ess_pmin, cap_trans):
        self.cap_max = np.concatenate((pmax, cap_trans, ess_pmax))
        self.cap_min = np.concatenate((pmin, np.zeros(nl), ess_pmin))
        return(self.cap_max, self.cap_min)

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

    def updateSOC(self, ng, nl, current_cap, ess_pmax, ess_duration, ess_socmax, ess_socmin, SOC_old):
        self.ess_emax = np.multiply(current_cap["max"][ng + nl::], ess_duration)
        self.ess_smax = np.multiply(self.ess_emax, ess_socmax)
        self.ess_smin = np.multiply(self.ess_emax, ess_socmin)
        SOC_old = current_cap["max"][ng + nl::]*SOC_old/ess_pmax
        return(self.ess_smax, self.ess_smin, SOC_old)

    def WindPower(self, nz, w_sites, zone_no, w_classes, r_cap, current_w_class, tr_mats, p_class, w_turbines, out_curve2, out_curve3):
        self.W = np.random.uniform(0, 1, (w_sites, w_classes))
        tr_mats[tr_mats == 0] = 1e-10
        self.time_wind = np.zeros((w_sites, w_classes))
        for w in range(w_sites):
            for c in range(w_classes):
                if current_w_class[w] == c:
                    self.time_wind[w, :] = -np.log(self.W[w, :])/tr_mats[w, c, :]
                    break
        temp = np.matrix(self.time_wind)
        tmin_wind = temp.argmin(1)
        current_w_class = tmin_wind
        self.w_power = np.zeros(w_sites)
        for w in range(w_sites):
            idx = int(tmin_wind[w].item())
            if p_class[w] == 2:
                self.w_power[w] = out_curve2[idx]*w_turbines[w]*r_cap[w]
            else:
                self.w_power[w] = out_curve3[idx]*w_turbines[w]*r_cap[w]
        self.w_zones = np.zeros(nz)
        for b in range(nz):
            for z in range(len(zone_no)):
                if b == zone_no[z] - 1:
                    self.w_zones[b] += self.w_power[z]
        return(self.w_zones, current_w_class)

    def SolarPower(self, n, nz, s_zone_no, solar_prob, s_profiles, s_sites, s_max):
        if n%24 == 0:
            self.month = np.floor(n/731).astype(int)
            self.prob_col = solar_prob[:, self.month]
            self.prob_index = np.array(list(zip(self.prob_col, range(len(self.prob_col)))))
            self.sorted_prob = self.prob_index[self.prob_index[:, 0].argsort()]
            self.sorted_prob[:, 0] = np.cumsum(self.sorted_prob[:, 0])
            self.rand_clust = np.random.uniform(0, 1)
            self.clust = 0
            for i in range(len(self.sorted_prob)):
                if i == 0 and self.rand_clust < self.sorted_prob[i, 0]:
                    self.clust = int(self.sorted_prob[i, 1])
                    break
                elif i > 0 and self.sorted_prob[i - 1, 0] < self.rand_clust < self.sorted_prob[i, 0]:
                    self.clust = int(self.sorted_prob[i, 1])
                    break
            self.solar_dim = s_profiles[self.clust].shape
            self.days = self.solar_dim[0]
            self.rand_day = np.floor(np.random.uniform(0, 1)*self.days).astype(int)
            sgen_sites = np.zeros((s_sites, 24))
            for sg in range(s_sites):
                sgen_sites[sg] = s_profiles[self.clust][self.rand_day, :, sg]*s_max[sg]
            self.s_zones = np.zeros((nz, 24))
            for b in range(nz):
                for z in range(len(s_zone_no)):
                    if b == s_zone_no[z] - 1:
                        self.s_zones[b] += sgen_sites[z]
        return(np.transpose(self.s_zones))

# Setup dummy parameters
ng = 10
nl = 5
ness = 3
nz = 3
w_sites = 4
w_classes = 5
s_sites = 2

# Initialize classes
py_raut = PythonRAUtilities()
import sys
sys.path.insert(0, "/Users/user/projects/QuESt/snl-progress")
from progress.mod_utilities import RAUtilities as RustRAUtilities
rust_raut = RustRAUtilities()

# 1. Benchmarking NextState
print("--- Benchmarking NextState (10,000 iterations) ---")
lambda_tot = np.random.uniform(0.01, 0.05, ng + nl + ness)
mu_tot = np.random.uniform(0.1, 0.5, ng + nl + ness)
current_state_py = np.ones(ng + nl + ness)
current_state_rust = np.ones(ng + nl + ness)
cap_max = np.random.uniform(50.0, 100.0, ng + nl + ness)
cap_min = np.random.uniform(0.0, 10.0, ng + nl + ness)
ess_units = np.array([2.0] * ness)

t0 = time.perf_counter()
t_min_py = 0.0
for _ in range(10000):
    current_state_py, current_cap_py, t_min_py = py_raut.NextState(
        t_min_py, ng, ness, nl, lambda_tot, mu_tot, current_state_py, cap_max, cap_min, ess_units
    )
py_time = time.perf_counter() - t0

t0 = time.perf_counter()
t_min_rust = 0.0
for _ in range(10000):
    current_state_rust, current_cap_rust, t_min_rust = rust_raut.NextState(
        t_min_rust, ng, ness, nl, lambda_tot, mu_tot, current_state_rust, cap_max, cap_min, ess_units
    )
rust_time = time.perf_counter() - t0
print(f"Python: {py_time:.4f}s")
print(f"Rust:   {rust_time:.4f}s (Speedup: {py_time/rust_time:.1f}x)")

# 2. Benchmarking updateSOC
print("\n--- Benchmarking updateSOC (10,000 iterations) ---")
ess_pmax = np.random.uniform(10.0, 20.0, ness)
ess_duration = np.random.uniform(4.0, 6.0, ness)
ess_socmax = np.array([0.9] * ness)
ess_socmin = np.array([0.1] * ness)
SOC_old_py = np.random.uniform(5.0, 15.0, ness)
SOC_old_rust = SOC_old_py.copy()

t0 = time.perf_counter()
for _ in range(10000):
    py_raut.updateSOC(ng, nl, current_cap_py, ess_pmax, ess_duration, ess_socmax, ess_socmin, SOC_old_py)
py_time = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(10000):
    rust_raut.updateSOC(ng, nl, current_cap_rust, ess_pmax, ess_duration, ess_socmax, ess_socmin, SOC_old_rust)
rust_time = time.perf_counter() - t0
print(f"Python: {py_time:.4f}s")
print(f"Rust:   {rust_time:.4f}s (Speedup: {py_time/rust_time:.1f}x)")

# 3. Benchmarking WindPower
print("\n--- Benchmarking WindPower (5,000 iterations) ---")
zone_no = np.array([1, 2, 1, 3], dtype=np.int64)
r_cap = np.random.uniform(45.0, 70.0, w_sites)
current_w_class_py = np.array([2, 1, 3, 0], dtype=np.int64)
current_w_class_rust = current_w_class_py.copy()
tr_mats_py = np.random.uniform(0.1, 0.5, (w_sites, w_classes, w_classes))
tr_mats_rust = tr_mats_py.copy()
p_class = np.array([2, 3, 2, 3], dtype=np.int64)
w_turbines = np.random.uniform(8.0, 15.0, w_sites)
out_curve2 = np.random.uniform(0.1, 1.0, w_classes)
out_curve3 = np.random.uniform(0.1, 1.0, w_classes)

t0 = time.perf_counter()
for _ in range(5000):
    py_raut.WindPower(nz, w_sites, zone_no, w_classes, r_cap, current_w_class_py, tr_mats_py, p_class, w_turbines, out_curve2, out_curve3)
py_time = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(5000):
    rust_raut.WindPower(nz, w_sites, zone_no, w_classes, r_cap, current_w_class_rust, tr_mats_rust, p_class, w_turbines, out_curve2, out_curve3)
rust_time = time.perf_counter() - t0
print(f"Python: {py_time:.4f}s")
print(f"Rust:   {rust_time:.4f}s (Speedup: {py_time/rust_time:.1f}x)")

# 4. Benchmarking SolarPower
print("\n--- Benchmarking SolarPower (5,000 iterations) ---")
s_zone_no = np.array([1, 2], dtype=np.int64)
solar_prob = np.random.uniform(0.1, 0.8, (2, 12))
profile_c0 = np.ones((5, 24, s_sites)) * 0.8
profile_c1 = np.ones((4, 24, s_sites)) * 0.4
s_profiles = [profile_c0, profile_c1]
s_max = np.array([100.0, 150.0])

t0 = time.perf_counter()
for h in range(5000):
    py_raut.SolarPower(h, nz, s_zone_no, solar_prob, s_profiles, s_sites, s_max)
py_time = time.perf_counter() - t0

t0 = time.perf_counter()
for h in range(5000):
    rust_raut.SolarPower(h, nz, s_zone_no, solar_prob, s_profiles, s_sites, s_max)
rust_time = time.perf_counter() - t0
print(f"Python: {py_time:.4f}s")
print(f"Rust:   {rust_time:.4f}s (Speedup: {py_time/rust_time:.1f}x)")
