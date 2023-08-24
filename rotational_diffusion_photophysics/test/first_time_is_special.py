import numpy as np
from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states
from rotational_diffusion_photophysics.models.detection import PolarizedDetection
from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion
from rotational_diffusion_photophysics.engine import System
import matplotlib.pyplot as plt


numerical_aperture = 1.4 # 1.4
refractive_index = 1.518 # 1.518
lmax = 6

### Create the detectors
detXY = PolarizedDetection(polarization=['xy'],
                            numerical_aperture=numerical_aperture,
                            refractive_index=refractive_index,
                            )

# Diffusion model
iso100us = IsotropicDiffusion(diffusion_coefficient=1/(6*100e-6))
iso1000us = IsotropicDiffusion(diffusion_coefficient=1/(6*1000e-6))

slow_lasers = ModulatedLasers(power_density=[100, 100],
                                  wavelength=[405, 488],
                                  polarization=['xy', 'xy'],
                                  modulation=[[0,1,0,1,0,1],[1,0,1,0,1,0]],
                                  time_windows=[1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1],
                                  time0=0,
                                  numerical_aperture=numerical_aperture,
                                  refractive_index=refractive_index,
                                  )

# STARSS modality 1
system = System(illumination=slow_lasers,
                fluorophore=rsEGFP2_8states,
                diffusion=iso100us,
                detection=detXY,
                lmax=lmax)


t = np.linspace(0,6e-1,1000)
signals = system.detector_signals(t)
pop = system._p
plt.plot(t, pop.transpose())
plt.legend(np.arange(system._p.shape[0]))
plt.xlabel('Time (s)')
plt.ylabel('Population')
plt.show()