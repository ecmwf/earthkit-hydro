Missing value handling philosophy
==================================

This page explains earthkit-hydro's approach to missing values and why it differs from some other hydrological tools.

The NaN convention
------------------

earthkit-hydro uses NumPy's ``np.nan`` (Not a Number) to represent missing values. This is a deliberate design choice that aligns with the scientific Python ecosystem and provides clear, predictable behavior.

**Key principle:** Any operation involving a missing value returns a missing value.

This is known as **NaN propagation** and is fundamental to how earthkit-hydro handles uncertainty.

Why NaN propagation?
--------------------

NaN propagation ensures that missing or invalid data doesn't silently corrupt your results:

- **Transparency:** NaN propagation makes missing data visible
- **Safety:** Prevents silent errors from ignored missing data
- **Consistency:** Same rules for all operations
- **Ecosystem:** Compatible with pandas, xarray, NumPy conventions
