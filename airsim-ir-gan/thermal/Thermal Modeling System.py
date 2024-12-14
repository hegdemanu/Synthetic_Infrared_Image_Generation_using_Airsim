import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MaterialProperties:
    """Properties of materials for thermal simulation."""
    emissivity: float
    specific_heat: float
    thermal_conductivity: float
    density: float

class ThermalModel:
    """
    Handles thermal calculations and material properties for IR simulation.
    """
    
    def __init__(self):
        """Initialize thermal model with default material properties."""
        self.materials = {
            'soil': MaterialProperties(0.914, 800, 0.52, 1500),
            'elephant': MaterialProperties(0.96, 3500, 0.6, 1000),
            'zebra': MaterialProperties(0.98, 3500, 0.6, 1000),
            'rhinoceros': MaterialProperties(0.96, 3500, 0.6, 1000),
            'hippopotamus': MaterialProperties(0.96, 3500, 0.6, 1000),
            'crocodile': MaterialProperties(0.96, 3500, 0.6, 1000),
            'human': MaterialProperties(0.985, 3500, 0.6, 1000),
            'tree': MaterialProperties(0.952, 2500, 0.12, 500),
            'grass': MaterialProperties(0.958, 2500, 0.12, 500),
            'shrub': MaterialProperties(0.986, 2500, 0.12, 500),
            'truck': MaterialProperties(0.8, 450, 45, 7800),
            'water': MaterialProperties(0.96, 4186, 0.6, 1000)
        }
        
        # Planck constants
        self.c1 = 1.19104e8  # First radiation constant
        self.c2 = 1.43879e4  # Second radiation constant
        
    def calculate_radiation(self, 
                          temperature: float,
                          emissivity: float,
                          wavelength_range: Tuple[float, float] = (8, 14),
                          dx: float = 0.01) -> Tuple[np.ndarray, float]:
        """
        Calculate spectral radiance using Planck's law.
        
        Args:
            temperature: Temperature in Celsius
            emissivity: Material emissivity (0-1)
            wavelength_range: (min, max) wavelength in microns
            dx: Wavelength step size
            
        Returns:
            Tuple of (spectral radiance array, integrated radiance)
        """
        # Convert to Kelvin
        temp_k = temperature + 273.15
        
        # Generate wavelength array
        wavelengths = np.arange(wavelength_range[0], wavelength_range[1], dx)
        
        # Calculate spectral radiance
        radiance = emissivity * (self.c1 / (wavelengths**5 * 
                               (np.exp(self.c2 / (wavelengths * temp_k)) - 1)))
        
        # Integrate over wavelength
        integrated = np.trapz(radiance, dx=dx)
        
        return radiance, integrated
    
    def calculate_temperature_distribution(self,
                                        ambient_temp: float,
                                        material: str,
                                        solar_radiation: float = 0,
                                        wind_speed: float = 0,
                                        time_step: float = 1.0) -> float:
        """
        Calculate surface temperature distribution considering environmental factors.
        
        Args:
            ambient_temp: Ambient temperature in Celsius
            material: Material name
            solar_radiation: Incident solar radiation (W/m²)
            wind_speed: Wind speed (m/s)
            time_step: Simulation time step (s)
            
        Returns:
            Surface temperature in Celsius
        """
        props = self.materials[material]
        
        # Heat transfer coefficients
        h_convection = 10.45 - wind_speed + 10 * np.sqrt(wind_speed)  # Simplified convection
        h_radiation = props.emissivity * 5.67e-8 * ((ambient_temp + 273.15)**4)  # Stefan-Boltzmann
        
        # Heat fluxes
        q_solar = solar_radiation * (1 - props.emissivity)  # Absorbed solar radiation
        q_convection = h_convection * (ambient_temp - ambient_temp)  # Initial temp = ambient
        q_radiation = h_radiation
        
        # Net heat flux
        q_net = q_solar - q_convection - q_radiation
        
        # Temperature change
        dT = q_net * time_step / (props.specific_heat * props.density)
        
        return ambient_temp + dT
    
    def convert_radiance_to_intensity(self,
                                    radiance: float,
                                    max_radiance: Optional[float] = None,
                                    min_radiance: Optional[float] = None) -> int:
        """
        Convert radiance value to 8-bit intensity value.
        
        Args:
            radiance: Radiance value
            max_radiance: Maximum radiance for normalization
            min_radiance: Minimum radiance for normalization
            
        Returns:
            8-bit intensity value (0-255)
        """
        if max_radiance is None:
            max_radiance = radiance * 1.2  # Arbitrary scaling
        if min_radiance is None:
            min_radiance = 0
            
        # Normalize and scale to 8-bit
        intensity = 255 * (radiance - min_radiance) / (max_radiance - min_radiance)
        return int(np.clip(intensity, 0, 255))
