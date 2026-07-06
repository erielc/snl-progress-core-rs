import numpy as np

try:
    from snl_progress_core_rs import RAUtilities as RustRAUtilities
    print("Successfully imported snl_progress_core_rs.RAUtilities!")
except ImportError as e:
    print("Could not import snl_progress_core_rs. Did you run 'maturin develop --release'?")
    raise e

nz = 3
s_sites = 2

s_zone_no = np.array([1, 2], dtype=np.int64) # site 0: zone 1, site 1: zone 2
# solar_prob has shape (num_clusters, 12). Let's make 2 clusters
solar_prob = np.array([
    [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # cluster 0
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # cluster 1
], dtype=np.float64)

# s_profiles is a list of 3D arrays of shape (days, 24, s_sites)
# Let's create mock profiles for 2 clusters
days_c0 = 5
days_c1 = 4
profile_c0 = np.ones((days_c0, 24, s_sites), dtype=np.float64) * 0.8
profile_c1 = np.ones((days_c1, 24, s_sites), dtype=np.float64) * 0.4

s_profiles = [profile_c0, profile_c1]
s_max = np.array([100.0, 150.0], dtype=np.float64)

rust_raut = RustRAUtilities()

# Test call 1: hour n = 0 (triggers calculation and caching)
s_zones_n0 = rust_raut.SolarPower(0, nz, s_zone_no, solar_prob, s_profiles, s_sites, s_max)

# Assertions
assert s_zones_n0.shape == (24, nz)
assert rust_raut.s_zones.shape == (nz, 24)

# Check that the aggregate sum matches max capacity * profile value:
# Since profile c0 is 0.8 and c1 is 0.4, the resulting zone values at any hour h should be:
# If cluster 0 selected: zone 0 = 0.8 * 100 = 80, zone 1 = 0.8 * 150 = 120, zone 2 = 0
# If cluster 1 selected: zone 0 = 0.4 * 100 = 40, zone 1 = 0.4 * 150 = 60, zone 2 = 0
val_zone0 = s_zones_n0[0, 0]
assert val_zone0 in [80.0, 40.0], f"Unexpected zone 0 value: {val_zone0}"

# Test call 2: hour n = 1 (uses cached s_zones, does not recalculate)
s_zones_n1 = rust_raut.SolarPower(1, nz, s_zone_no, solar_prob, s_profiles, s_sites, s_max)
np.testing.assert_allclose(s_zones_n0, s_zones_n1)

print("SolarPower verification tests passed successfully!")
