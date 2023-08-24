[![DOI](https://zenodo.org/badge/DOI/10.1038/s41587-022-01489-7.svg)](https://doi.org/10.1038/s41587-022-01489-7)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7009614.svg)](https://doi.org/10.5281/zenodo.7009614)

# rotational_diffusion_photophysics
Tools for computing time-dependent fluorescence signals with rotational diffusion and complex photophysics.  
An arbitrary kinetic scheme can be used and the program analytically solves the diffusion-kinetics problem. All the angular probability distribution functions are expanded with spherical harmonics.  
This repository was used in the scientific paper [A. Volpato et al. Nat Biotechnol 41, 552–559 (2023)](https://doi.org/10.1038/s41587-022-01489-7).

## Module content
The pacakge is usually imported as `import rotational_diffusion_photophyscis as rdp`

- `rdp.engine`  
Main engine of the kinetics and diffusion solver for an arbitrary kinetics scheme and isotorpic free rotatioal diffusion.

- `rdp.models`  
Several models to create a full system class `rdp.engine.System`:
  - `rdp.models.illumination`  
  Classes with laser modulation
  - `rdp.models.detection`  
  Classes for detection. Currently only `PolarizedDetection`
  - `rdp.models.diffusion`  
  Classes with rotational diffusion models
  - `rdp.models.fluorophore`  
  Fluorophore models, including `NegativeSwitcher` and `STEDDye`
  - `rdp.models.starss`  
  Classes for STARSS experiments. A few `System` classes are defined.

- `rdp.plot`  
Some useful plotting tools for orientational probabilities and STARSS pulse schemes.


## Additional content

- `starss` folder  
Notebooks with computation and plotting of example STARSS simulations.

- `notes` folder  
Mathematical notes about the rotational diffusion and kinetics model.


## Install
Using pip and making a symlink.
From the main folder launch: `pip install -e .`