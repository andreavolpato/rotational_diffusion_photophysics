import numpy as np

wavelength = [365, 405, 488, 560, 600, 640, 680, 700, 775]
laser = [488, 488, 775, 405]

def find_wavelenght(wavelength, laser):
    wavelength = np.array(wavelength)
    laser = np.array(laser)

    wavelength_to_use = np.isin(wavelength, laser)
    laser_to_use = np.isin(laser, wavelength)
    assert np.all(laser_to_use), "Fluorophore data at one or more laser wavelenghts is missing."

    wavelength_indexes = np.zeros(laser.shape)
    for i, laseri in enumerate(laser):
        wavelength_indexes[i] = np.where(wavelength == laseri)[0]
    wavelength_indexes = np.int32(wavelength_indexes)
    return wavelength_indexes

print(find_wavelenght(wavelength, laser))
