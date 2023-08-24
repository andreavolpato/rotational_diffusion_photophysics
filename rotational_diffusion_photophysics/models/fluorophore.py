import numpy as np

################################################################################
# Reversibly switchable fluorescence proteins (RSFP) classes
################################################################################

class NegativeSwitcher:
    def __init__(self,
                 extinction_coeff_on=[0, 0],
                 extinction_coeff_off=[0, 0],
                 wavelength=[405, 488],
                 lifetime_on=3e-9,
                 lifetime_off=16e-12,
                 quantum_yield_on_to_off=0.001,
                 quantum_yield_off_to_on=0.2,
                 quantum_yield_on_fluo=1,
                 starting_populations=[1,0,0,0],
                 deprotonation_time_off = 15e-6,  # for 6 states model
                 protonation_time_on = 150e-6,  # for 6 states model
                 quantum_yield_trans_to_cis_anionic=0,  # for 8 states model
                 quantum_yield_cis_to_trans_neutral=0, # for 8 states model
                 quantum_yield_cis_anionic_bleaching=0,
                 quantum_yield_trans_anionic_bleaching=0,
                 nspecies=4):
        # Cross section in cm2 of absorptions
        epsilon2sigma = 3.825e-21  # [Tkachenko2007, page 5]
        self.extinction_coeff_on = np.array(extinction_coeff_on)  # [M-1 cm-1]
        self.extinction_coeff_off = np.array(extinction_coeff_off)  # [M-1 cm-1]
        self.cross_section_on = self.extinction_coeff_on * epsilon2sigma  # [cm2]
        self.cross_section_off = self.extinction_coeff_off * epsilon2sigma  # [cm2]
        self.wavelength = np.array(wavelength)  # [nm]

        # Lifetime of the on excited state in seconds
        # Assumption: lifetime on and off are the same for the same protonation
        # state. This might not be the case, especially because cis_neutral
        # species is not fluorescent, so most likely it will have a shorter
        # lifetime. For small enough excitation kinetic rates, if we don't have
        # accumulation of exited state species, then it doesn't matter much.
        self.lifetime_on = lifetime_on  # [s]
        self.lifetime_off = lifetime_off  # [s]

        # Quantum yield of an off-switching event from the on excited state and
        # fluorescence from the on state
        self.quantum_yield_on_fluo = quantum_yield_on_fluo
        self.quantum_yield_on_to_off = quantum_yield_on_to_off  # cis_to_trans_anionic 
        self.quantum_yield_off_to_on = quantum_yield_off_to_on  # trans_to_cis_neutra

        # Quantum yeilds and Protonation and deprotonation times for 6 and 8
        # states models.
        self.quantum_yield_cis_to_trans_neutral = quantum_yield_cis_to_trans_neutral
        self.quantum_yield_trans_to_cis_anionic = quantum_yield_trans_to_cis_anionic
        self.protonation_time_on = protonation_time_on
        self.deprotonation_time_off = deprotonation_time_off

        # Bleaching quantum yields, assuming only a bleaching channel from 
        # excitaiton of the on state.
        self.quantum_yield_cis_anionic_bleaching = quantum_yield_cis_anionic_bleaching
        self.quantum_yield_trans_anionic_bleaching = quantum_yield_trans_anionic_bleaching

        # Label describing the fluorophore type
        # Number of states in the kinetic model
        self.nspecies = nspecies

        # Index of the fluorescent state
        # Here, only one fluorescent state is assumed.
        # This is not necessary in general and could be extended in the future.
        self.quantum_yield_fluo = np.zeros((nspecies))
        self.quantum_yield_fluo[1] = self.quantum_yield_on_fluo

        # Population at the beginning of the experiment
        self.starting_populations = starting_populations
        return None

    def kinetics_matrix(self, l, m, F, wavelength_laser):
        # Compute the kinetics matrix expanded in l, m coefficients
        # Note that the order of the laser in F must be consistent with the
        # choices made in the fluorophore class.
        # In this case F[0] is the blue light.

        # l, m, F, and wavelength must be provided by outside.
        # l and m are the quantum numbers of SH, while F encodes the info about
        # photon flux and it's angular dependence.

        #TODO remove dependence from l, m, F and add them in a generic way after

        # Initialize arrays
        Feye = np.eye( (l.size) )
        K = np.zeros( (self.nspecies, self.nspecies, l.size, l.size))

        # Put the kinetic constants connecting the right species
        # K[1,0] is the kinetic constant for the proces 1 <- 0.
        nlasers = F.shape[0]
        nwavelengths = self.wavelength.size
        if self.nspecies == 4:
            self.fluorophore_type = 'rsFP_negative_4states'
            for i in np.arange(nlasers):
                for j in np.arange(nwavelengths):
                    if wavelength_laser[i] == self.wavelength[j]:
                        K[1,0] = K[1,0] + F[i] * self.cross_section_on[j]
                        K[3,2] = K[3,2] + F[i] * self.cross_section_off[j]
            K[0,1] = Feye / self.lifetime_on
            K[2,1] = Feye / self.lifetime_on  * self.quantum_yield_on_to_off
            K[2,3] = Feye / self.lifetime_off
            K[0,3] = Feye / self.lifetime_off * self.quantum_yield_off_to_on
        
        if self.nspecies == 6:
            self.fluorophore_type = 'rsFP_negative_6states'
            for i in np.arange(nlasers):
                for j in np.arange(nwavelengths):
                    if wavelength_laser[i] == self.wavelength[j]:
                        K[1,0] = K[1,0] + F[i] * self.cross_section_on[j]
                        K[4,3] = K[4,3] + F[i] * self.cross_section_off[j]
            K[0,1] = Feye / self.lifetime_on
            K[2,1] = Feye / self.lifetime_on  * self.quantum_yield_on_to_off
            K[3,2] = Feye / self.protonation_time_on
            K[3,4] = Feye / self.lifetime_off
            K[5,4] = Feye / self.lifetime_off * self.quantum_yield_off_to_on
            K[0,5] = Feye / self.deprotonation_time_off
        
        if self.nspecies == 8:
            self.fluorophore_type = 'rsFP_negative_8states'
            for i in np.arange(nlasers):
                for j in np.arange(nwavelengths):
                    if wavelength_laser[i] == self.wavelength[j]:
                        K[1,0] = K[1,0] + F[i] * self.cross_section_on[j]  # cis anionic
                        K[3,2] = K[3,2] + F[i] * self.cross_section_on[j]  # trans anionic
                        K[5,4] = K[5,4] + F[i] * self.cross_section_off[j]  # trans neutral
                        K[7,6] = K[7,6] + F[i] * self.cross_section_off[j]  # cis neutral
            # On-branch
            K[0,1] = Feye / self.lifetime_on
            K[2,1] = Feye / self.lifetime_on  * self.quantum_yield_on_to_off
            K[2,3] = Feye / self.lifetime_on
            K[0,3] = Feye / self.lifetime_on  * self.quantum_yield_trans_to_cis_anionic
            K[4,2] = Feye / self.protonation_time_on

            # Off-branch
            K[4,5] = Feye / self.lifetime_off
            K[6,5] = Feye / self.lifetime_off * self.quantum_yield_off_to_on
            K[6,7] = Feye / self.lifetime_off
            K[4,7] = Feye / self.lifetime_off * self.quantum_yield_cis_to_trans_neutral
            K[0,6] = Feye / self.deprotonation_time_off
        
        if self.nspecies == 9:
            self.fluorophore_type = 'rsFP_negative_9states_bleaching'
            for i in np.arange(nlasers):
                for j in np.arange(nwavelengths):
                    if wavelength_laser[i] == self.wavelength[j]:
                        K[1,0] = K[1,0] + F[i] * self.cross_section_on[j]  # cis anionic
                        K[3,2] = K[3,2] + F[i] * self.cross_section_on[j]  # trans anionic
                        K[5,4] = K[5,4] + F[i] * self.cross_section_off[j]  # trans neutral
                        K[7,6] = K[7,6] + F[i] * self.cross_section_off[j]  # cis neutral
            # On-branch
            K[0,1] = Feye / self.lifetime_on
            K[2,1] = Feye / self.lifetime_on  * self.quantum_yield_on_to_off
            K[2,3] = Feye / self.lifetime_on
            K[0,3] = Feye / self.lifetime_on  * self.quantum_yield_trans_to_cis_anionic
            K[4,2] = Feye / self.protonation_time_on

            # Off-branch
            K[4,5] = Feye / self.lifetime_off
            K[6,5] = Feye / self.lifetime_off * self.quantum_yield_off_to_on
            K[6,7] = Feye / self.lifetime_off
            K[4,7] = Feye / self.lifetime_off * self.quantum_yield_cis_to_trans_neutral
            K[0,6] = Feye / self.deprotonation_time_off

            # Bleaching channels from teh on-branch
            K[8,1] = Feye / self.lifetime_on * self.quantum_yield_cis_anionic_bleaching
            K[8,3] = Feye / self.lifetime_on * self.quantum_yield_trans_anionic_bleaching
        return K

    def starting_coeffs(self, l, m):
        # Compute the starting values for the population SH coefficients
        c0 = np.zeros( (self.nspecies, l.size) )
        c0[:,0] = self.starting_populations
        return c0

# rsEGFP2 Models
rsEGFP2_4states = NegativeSwitcher(extinction_coeff_on=[5260, 51560],
                                    extinction_coeff_off=[22000, 60],
                                    wavelength=[405, 488],
                                    lifetime_on=1.6e-9,
                                    quantum_yield_on_fluo=0.35,
                                    quantum_yield_on_to_off=1.65e-2,
                                    )

rsEGFP2_8states = NegativeSwitcher(extinction_coeff_on=  [  5260, 51560],
                                    extinction_coeff_off=[ 22000,    60],
                                    wavelength=          [   405,   488],
                                    lifetime_on=1.6e-9,
                                    lifetime_off=20e-12,
                                    quantum_yield_on_to_off=1.65e-2,
                                    quantum_yield_off_to_on=0.33,
                                    quantum_yield_on_fluo=0.35,
                                    starting_populations=[1,0,0,0,0,0,0,0],
                                    deprotonation_time_off=5.1e-6,
                                    protonation_time_on=48e-6,
                                    nspecies=8,
                                    quantum_yield_trans_to_cis_anionic=0.0165,
                                    quantum_yield_cis_to_trans_neutral=0.33,
                                    )

rsEGFP2_9states = NegativeSwitcher(extinction_coeff_on=  [  5260, 51560],
                                    extinction_coeff_off=[ 22000,    60],
                                    wavelength=          [   405,   488],
                                    lifetime_on=1.6e-9,
                                    lifetime_off=20e-12,
                                    quantum_yield_on_to_off=1.65e-2,
                                    quantum_yield_off_to_on=0.33,
                                    quantum_yield_on_fluo=0.35,
                                    starting_populations=[1,0,0,0,0,0,0,0,0],
                                    deprotonation_time_off=5.1e-6,
                                    protonation_time_on=48e-6,
                                    nspecies=9,
                                    quantum_yield_trans_to_cis_anionic=0.0165,
                                    quantum_yield_cis_to_trans_neutral=0.33,
                                    )


################################################################################
# STED dye classes
################################################################################

class STEDDye:
    def __init__(self,
                 extinction_coeff_exc=[0, 0],
                 extinction_coeff_sted=[0, 0],
                 wavelength=[640, 775],
                 lifetime=3e-9,
                 quantum_yield_fluo=1,
                 starting_populations=[1,0],
                 nspecies=2):
        # Cross section in cm2 of absorptions
        epsilon2sigma = 3.825e-21  # [Tkachenko2007, page 5]
        self.extinction_coeff_exc = np.array(extinction_coeff_exc)  # [M-1 cm-1]
        self.extinction_coeff_sted = np.array(extinction_coeff_sted)  # [M-1 cm-1]
        self.cross_section_exc = self.extinction_coeff_exc * epsilon2sigma  # [cm2]
        self.cross_section_sted = self.extinction_coeff_sted * epsilon2sigma  # [cm2]
        self.wavelength = np.array(wavelength)  # [nm]

        # Lifetime of the on excited state in seconds
        self.lifetime = lifetime  # [s]


        self.quantum_yield_fluo = quantum_yield_fluo

        # Label describing the fluorophore type
        # Number of states in the kinetic model
        self.nspecies = nspecies

        # Quantum yield of fluorescence
        # Here, only one fluorescent state is assumed.
        # This is not necessary in general and could be extended in the future.
        self.quantum_yield_fluo = np.zeros((nspecies))
        self.quantum_yield_fluo[1] = quantum_yield_fluo

        # Population at the beginning of the experiment
        self.starting_populations = starting_populations

    def kinetics_matrix(self, l, m, F, wavelength_laser):
        # Compute the kinetics matrix expanded in l, m coefficients
        # Note that the order of the laser in F must be consistent with the
        # choices made in the fluorophore class.
        # In this case F[0] is the blue light.

        # l, m, F, and wavelength must be provided by outside.
        # l and m are the quantum numbers of SH, while F encodes the info about
        # photon flux and it's angular dependence.

        # Initialize arrays
        Feye = np.eye( (l.size) )
        K = np.zeros( (self.nspecies, self.nspecies, l.size, l.size))

        # Put the kinetic constants connecting the right species
        # K[1,0] is the kinetic constant for the proces 1 <- 0.
        nlasers = F.shape[0]
        nwavelengths = self.wavelength.size
        if self.nspecies == 2:
            self.fluorophore_type = 'rsFP_negative_4states'
            for i in np.arange(nlasers):
                for j in np.arange(nwavelengths):
                    if wavelength_laser[i] == self.wavelength[j]:
                        K[1,0] = K[1,0] + F[i] * self.cross_section_exc[j]
                        K[0,1] = K[0,1] + F[i] * self.cross_section_sted[j]
            K[0,1] = K[0,1] + Feye / self.lifetime
        return K

    def starting_coeffs(self, l, m):
        # Compute the starting values for the population SH coefficients
        c0 = np.zeros( (self.nspecies, l.size) )
        c0[:,0] = self.starting_populations
        return c0

# atto647N
# parameterization from: J. Oracz, V. Westphal, C. Radzewicz, S. J. Sahl, and S. W. Hell, Scientific Reports 7, 1 (2017)
# https://doi.org/10.1021/acs.nanolett.7b00468

atto647N = STEDDye(extinction_coeff_exc=[25950, 0],
                   extinction_coeff_sted=[0, 1250],
                   wavelength=[640, 775],
                   quantum_yield_fluo=0.65,
                   lifetime=3.5e-9,
                   )
