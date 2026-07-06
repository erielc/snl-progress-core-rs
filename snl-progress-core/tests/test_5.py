import numpy as np

try:
    from snl_progress_core_rs import RAUtilities as RustRAUtilities
    print("Successfully imported snl_progress_core_rs.RAUtilities!")
except ImportError as e:
    print("Could not import snl_progress_core_rs. Did you run 'maturin develop --release'?")
    raise e

nz = 3
w_sites = 4
w_classes = 5

zone_no = np.array([1, 2, 1, 3], dtype=np.int64) # site 0: zone 1, site 1: zone 2, site 2: zone 1, site 3: zone 3
r_cap = np.array([50.0, 60.0, 45.0, 70.0], dtype=np.float64)
current_w_class = np.array([2, 1, 3, 0], dtype=np.int64)
tr_mats = np.zeros((w_sites, w_classes, w_classes), dtype=np.float64)
# populate tr_mats with some mock transition rates
for s in range(w_sites):
    for c1 in range(w_classes):
        for c2 in range(w_classes):
            tr_mats[s, c1, c2] = 0.1 * (c1 + c2 + 1)

p_class = np.array([2, 3, 2, 3], dtype=np.int64)
w_turbines = np.array([10.0, 15.0, 8.0, 12.0], dtype=np.float64)
out_curve2 = np.array([0.1, 0.3, 0.6, 0.8, 1.0], dtype=np.float64)
out_curve3 = np.array([0.05, 0.25, 0.55, 0.75, 0.95], dtype=np.float64)

# Call Rust version
rust_raut = RustRAUtilities()
w_zones, current_w_class_new = rust_raut.WindPower(
    nz, w_sites, zone_no, w_classes, r_cap, current_w_class,
    tr_mats, p_class, w_turbines, out_curve2, out_curve3
)

# Assertions
assert len(w_zones) == nz
assert len(current_w_class_new) == w_sites

# Check properties on the Rust object
assert rust_raut.W.shape == (w_sites, w_classes)
assert rust_raut.time_wind.shape == (w_sites, w_classes)
assert len(rust_raut.w_power) == w_sites
assert len(rust_raut.w_zones) == nz

# Verify that current_w_class was mutated in-place
np.testing.assert_allclose(current_w_class, current_w_class_new)

# Manually verify power aggregation logic for the returned random state:
# Let's read the chosen classes
tmin_wind = current_w_class_new
expected_power = np.zeros(w_sites)
for w in range(w_sites):
    c_min = tmin_wind[w]
    if p_class[w] == 2:
        expected_power[w] = out_curve2[c_min] * w_turbines[w] * r_cap[w]
    else:
        expected_power[w] = out_curve3[c_min] * w_turbines[w] * r_cap[w]

np.testing.assert_allclose(rust_raut.w_power, expected_power)

expected_zones = np.zeros(nz)
for z in range(w_sites):
    expected_zones[zone_no[z] - 1] += expected_power[z]

np.testing.assert_allclose(w_zones, expected_zones)

print("WindPower verification tests passed successfully!")
