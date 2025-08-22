# NASA ARSET Training Materials

## NEX-GDDP-CMIP6 Intake Catalog using OpenVisus

This repository provides a custom [Intake](https://intake.readthedocs.io/) data source for loading and subsetting the NASA NEX-GDDP-CMIP6 dataset using [OpenVisus](https://github.com/sci-visus/OpenVisus). It allows easy access to daily downscaled climate projections with support for:

- Custom `model`, `variable`, `scenario`, and `timestamp`
- Downsampled resolutions via `quality`
- Subsetting by latitude/longitude bounds (`lat_range`, `lon_range`)
- Output in `xarray.DataArray` format

---

##  Installation

Clone the repo and ensure the following dependencies are installed:

```bash
pip install intake xarray numpy
# and OpenVisus if not already installed:
pip install openvisus
```

---

## 🔧 Parameters

| Parameter     | Type   | Required | Description                                   |
|---------------|--------|----------|-----------------------------------------------|
| `model`       | str    | ✅        | CMIP6 model name (e.g. `ACCESS-CM2`)          |
| `variable`    | str    | ✅        | Variable name (e.g. `tas`, `pr`, `rhs`)       |
| `scenario`    | str    | ✅        | Emissions scenario (e.g. `historical`, `ssp585`) |
| `timestamp`   | str    | ✅        | Date in `YYYY-MM-DD` format                   |
| `quality`     | int    | ❌        | Resolution level (`0`=full, `-1`=half, etc.) default=0  |
| `lat_range`   | tuple  | ❌        | Latitude range `(min, max)` in degrees, default=entire region        |
| `lon_range`   | tuple  | ❌        | Longitude range `(min, max)` in degrees, default= entire region       |


--- 

## 🧪 Usage Example

```python
import intake

cat = intake.open_catalog("cmip6_catalog.yml")

ds = cat.nex_gddp_cmip6(
    model="CMCC-CM2-SR5",
    variable="tas",
    scenario="historical",
    timestamp="2005-06-15",
    quality=-2,
    lat_range=(0, 40),
    lon_range=(60, 120)
).read()

ds.plot()
```

---
## 🚀 Launch on Binder

Click the button below to launch this repository in an interactive Jupyter environment via [Binder](https://mybinder.org/):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aashishpanta0/nasa-arset-training/binder)

This will launch a temporary cloud environment where you can:
- Browse the Intake catalog
- Read and subset NEX-GDDP-CMIP6 climate data
- Plot data interactively using Jupyter Notebooks
---
---

## 🙏 Acknowledgements

This work was developed as part of ongoing research at the NASA JPL, [National Science Data Fabric(NSDF)](https://nationalsciencedatafabric.org/), and [Scientific Computing and Imaging (SCI) Institute](https://www.sci.utah.edu/) at the University of Utah. The library, data and workflow here is managed by [National Science Data Fabric(NSDF)](https://nationalsciencedatafabric.org/). 

<!-- This work is supported in par NSF OAC award 2138811, -->
--- 
## Developers

    Aashish Panta, Alex Goodman, Kyo Lee, Valerio Pascucci
    University of Utah, NASA JPL
