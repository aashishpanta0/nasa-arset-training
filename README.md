
# NASA ARSET Training Repository

[**Dashboard**](https://nasa-arset.nationalsciencedatafabric.org) | [**NSDF-ARSET**](https://nationalsciencedatafabric.org/nasa-arset) | [**Register for ARSET Training**](https://www.earthdata.nasa.gov/learn/trainings/assessing-extreme-weather-statistics-using-nasa-earth-exchange-global-daily)

---

## Overview

This repository provides materials and code for NASA ARSET training on assessing extreme weather statistics using NASA Earth Exchange Global Daily Downscaled Projections (NEX-GDDP-CMIP6). It includes:

- **Intake catalog** for NEX-GDDP-CMIP6 climate data
- **ETCCDI indices** calculation scripts and notebooks
- **Country-level analysis** using Natural Earth shapefiles
- **Interactive Jupyter notebooks** for data exploration and visualization

---


## Repository Structure

- `cmip6_catalog.yml` — Intake catalog for NEX-GDDP-CMIP6 data
- `requirements.txt` — Python dependencies
- `scripts/` — Python scripts and Jupyter notebooks:
    - ETCCDI indices drivers and calculators (`ETCCDI_*`)
    - Data download and streaming (`download_nex_gddp.py`, `Streaming_Via_Intake.ipynb`)
    - Plotting and country stats (`Plot_NEX-GDDP.ipynb`, `ETCCDI_country_stats.ipynb`)
    - Country boundaries (Natural Earth shapefiles)


---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aashishpanta0/nasa-arset-training.git
cd nasa-arset-training
pip install -r requirements.txt
```

---

## Usage

### Intake Catalog Example

```python
import intake
cat = intake.open_catalog("cmip6_catalog.yml")
ds = cat.nex_gddp_cmip6(
        model="CMCC-CM2-SR5",
        variable="tas",
        scenario="historical",
        timestamp="2005-06-15",
        quality=0
).read()
ds.plot()
```

### ETCCDI Indices & Country Analysis
- Use the Jupyter notebooks in `scripts/` for step-by-step workflows on climate indices and country-level statistics.
- Shapefiles for country boundaries are provided in `scripts/shapefile/` (source: [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/)).

---

## Data Sources

- **NEX-GDDP-CMIP6**: Daily downscaled climate projections
- **Natural Earth**: Country boundaries shapefiles

---

## Launch Interactive Environment

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aashishpanta0/nasa-arset-training/binder)

Launch a cloud Jupyter environment to browse the catalog, analyze data, and plot interactively.

---

## 🙏 Acknowledgements

This work was developed as part of ongoing research at the NASA JPL, [National Science Data Fabric(NSDF)](https://nationalsciencedatafabric.org/), and [Scientific Computing and Imaging (SCI) Institute](https://www.sci.utah.edu/) at the University of Utah. The library, data and workflow here is managed by [National Science Data Fabric(NSDF)](https://nationalsciencedatafabric.org/). 

<!-- This work is supported in par NSF OAC award 2138811, -->
--- 
## Developers

    Aashish Panta, Alex Goodman, Kyo Lee, Valerio Pascucci
    University of Utah, NASA JPL
