import numpy as np
from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states
from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states_pka
from rotational_diffusion_photophysics.models.detection import PolarizedDetection
from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion
from rotational_diffusion_photophysics.plot.plot_pulse_scheme import plot_pulse_scheme
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

dt = 5e-3
slow_lasers = ModulatedLasers(power_density=[200, 1000],
                                  wavelength=[405, 488],
                                  polarization=['xy', 'xy'],
                                  modulation=[[0,0,1,0,1],[0,1,0,1,0]],
                                  time_windows=np.ones((5))*dt,
                                  time0=0,
                                  numerical_aperture=numerical_aperture,
                                  refractive_index=refractive_index,
                                  )

system = System(illumination=slow_lasers,
                fluorophore=rsEGFP2_8states,
                diffusion=iso100us,
                detection=detXY,
                lmax=lmax)

system_pka = System(illumination=slow_lasers,
                fluorophore=rsEGFP2_8states_pka,
                diffusion=iso100us,
                detection=detXY,
                lmax=lmax)

plt.figure(1)
t = np.linspace(0,5*dt,1000)
signals = system.detector_signals(t)
pop = system._p
plt.plot(t, pop.transpose())
plt.legend(np.arange(pop.shape[0]))
plt.xlabel('Time (s)')
plt.ylabel('Population')
plt.title('pka << ph')
plt.savefig('populations.png')
    
plt.figure(2)
t = np.linspace(0,5*dt,1000)
signals = system_pka.detector_signals(t)
pop = system_pka._p
plt.plot(t, pop.transpose())
plt.legend(np.arange(pop.shape[0]))
plt.xlabel('Time (s)')
plt.ylabel('Population')
plt.title('pka=5.9, ph=7.5')
plt.savefig('populations_pka.png')

plt.figure(3)
plot_pulse_scheme(system, xlim=[0, 5*dt], ylim=[0, 1500])
plt.title('Pulse scheme')
plt.savefig('pulse_scheme.png')
plt.show()