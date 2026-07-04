>   1      Rust & PyO3 Step-by-Step Learning Guide
    2
    3     This guide will walk you through implementing the high-performance Rust
          dynamic library ( snl_progress_core_rs ) yourself.
    4
    5     For each step, we provide:

    6     • The Rust concepts you'll learn.
    7     • The exact function signature.
    8     • A code skeleton/stub with  TODO s.
    9     • Compilation/verification commands.
          ──────
   13     ## Step 1:  reltrates  (Reciprocal and Concatenation)
   14
   15     The goal is to concatenate three inputs (MTTF for generators,
          transmission, and ESS), take their element-wise reciprocal, and return
          them to Python.
   16
   17     ### Rust Concepts to Learn

   18     1. Iterators ( iter() ,  chain() ,  map() ,  copied() ):
   19     Rust standard library arrays/slices do not have a built-in  concatenate
          function like  numpy . Instead, we chain iterators together to form a
          single continuous stream and  collect  them into a new collection ( Vec<T>
          ).
   20     2. PyO3 Bound Lifetimes ( 'py  and  Bound<'py, T> ):
   21     Since Python objects live on Python's garbage-collected heap, we must
          associate any references to them with the lifetime of the Python GIL (
          Python<'py> ).
   22     3. Numpy FFI conversion ( to_pyarray ):
   23     To return data back to Python as a NumPy array, we allocate the array
          directly on the Python heap using  .to_pyarray(py) .
   24
   25     ### Code Skeleton to copy into  src/lib.rs
   26
   27       use pyo3::prelude::*;
   28       use pyo3::types::PyDict;
   29       use pyo3::PyObject;
   30       use numpy::{PyReadonlyArray1, ToPyArray, PyArray1};
   31       use rand::Rng;
   32
   33       #[pyclass]
   34       #[derive(Default)]
   35       pub struct RAUtilities {
   36           #[pyo3(get, set)]
   37           pub t_min: f64,
   38           #[pyo3(get, set)]
  [0%  L1  1-47/297]

  ↑/↓ scroll · pgup/pgdown page · shift+g bottom · g top · c comment
  l hide lines · esc close
? for shortcuts                                                Gemini 3.5 Flash (Medium)