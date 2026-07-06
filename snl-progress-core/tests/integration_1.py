import sys
sys.path.insert(0, "/Users/user/projects/QuESt/snl-progress")

try:
    from progress.mod_utilities import RAUtilities, _HAS_RUST
    print("Integration test:")
    print("  _HAS_RUST =", _HAS_RUST)
    raut = RAUtilities()
    # Check if the methods are present
    for m in ["reltrates", "capacities", "NextState", "updateSOC", "WindPower", "SolarPower"]:
        print(f"  hasattr({m})? {hasattr(raut, m)}")
    print("Integration check passed!")
except Exception as e:
    print("Error during integration test:", e)
    raise e
