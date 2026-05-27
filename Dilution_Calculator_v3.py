#!/usr/bin/env python3
"""
XRD Sample Dilution Calculator - Enhanced Streamlit Version
Calculate optimal powder dilution with comprehensive elemental data from local files
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import io


class AtomicDataManager:
    """
    Manages atomic scattering factor data from local .nff files.
    Provides mass attenuation coefficients for all elements at arbitrary energies.
    """
    
    ELEMENTS = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
        21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
        41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
        61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
        71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
        91: 'Pa', 92: 'U'
    }
    
    ATOMIC_WEIGHTS = {
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81, 'C': 12.01, 'N': 14.01, 'O': 16.00,
        'F': 19.00, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.31, 'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.07,
        'Cl': 35.45, 'Ar': 39.95, 'K': 39.10, 'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00,
        'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38, 'Ga': 69.72, 'Ge': 72.63,
        'As': 74.92, 'Se': 78.96, 'Br': 79.90, 'Kr': 83.80, 'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22,
        'Nb': 92.91, 'Mo': 95.96, 'Tc': 98.00, 'Ru': 101.1, 'Rh': 102.9, 'Pd': 106.4, 'Ag': 107.9, 'Cd': 112.4,
        'In': 114.8, 'Sn': 118.7, 'Sb': 121.8, 'Te': 127.6, 'I': 126.9, 'Xe': 131.3, 'Cs': 132.9, 'Ba': 137.3,
        'La': 138.9, 'Ce': 140.1, 'Pr': 140.9, 'Nd': 144.2, 'Pm': 145.0, 'Sm': 150.4, 'Eu': 152.0, 'Gd': 157.3,
        'Tb': 158.9, 'Dy': 162.5, 'Ho': 164.9, 'Er': 167.3, 'Tm': 168.9, 'Yb': 173.0, 'Lu': 175.0, 'Hf': 178.5,
        'Ta': 180.9, 'W': 183.8, 'Re': 186.2, 'Os': 190.2, 'Ir': 192.2, 'Pt': 195.1, 'Au': 197.0, 'Hg': 200.6,
        'Tl': 204.4, 'Pb': 207.2, 'Bi': 209.0, 'Po': 209.0, 'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0,
        'Ac': 227.0, 'Th': 232.0, 'Pa': 231.0, 'U': 238.0
    }
    
    def __init__(self, data_dir: str = './ScatteringFactors', cache_dir: str = './atomic_data_cache'):
        """
        Initialize the atomic data manager.
        
        Args:
            data_dir: Directory containing .nff files
            cache_dir: Directory for cached processed data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.data_cache = {}
        self.interpolators = {}
        self.element_map = {}  # Map to handle case variations
        
        # Load data from .nff files
        self._load_all_nff_files()
        
        # Try to load cached interpolators
        self._load_cache()
    
    def _normalize_element(self, element: str) -> str:
        """
        Normalize element symbol to standard format (e.g., 'Fe', 'Cu').
        Handles case variations.
        """
        # Check if already in standard form
        if element in self.ATOMIC_WEIGHTS:
            return element
        
        # Try to find in element map
        if element in self.element_map:
            return self.element_map[element]
        
        # Try capitalize
        normalized = element.capitalize()
        if normalized in self.ATOMIC_WEIGHTS:
            self.element_map[element] = normalized
            return normalized
        
        # Try upper
        normalized = element.upper()
        if normalized in self.ATOMIC_WEIGHTS:
            self.element_map[element] = normalized
            return normalized
        
        # Try lower then capitalize
        normalized = element.lower().capitalize()
        if normalized in self.ATOMIC_WEIGHTS:
            self.element_map[element] = normalized
            return normalized
        
        return element  # Return as-is if not found
    def _validate_and_clean_data(self, element: str) -> bool:
        """
        Validate and clean data for an element.
        
        Args:
            element: Element symbol
            
        Returns:
            True if data is valid, False otherwise
        """
        if element not in self.data_cache:
            return False
        
        data = self.data_cache[element]
        
        try:
            energies = data['energy']
            mu_rho = data['mu_rho']
            
            # Check for sufficient data points
            if len(energies) < 4:
                print(f"Removing {element}: insufficient data points ({len(energies)})")
                del self.data_cache[element]
                return False
            
            # Check for monotonicity after sorting
            sort_idx = np.argsort(energies)
            sorted_energies = energies[sort_idx]
            
            # Check energy range
            if sorted_energies[-1] / sorted_energies[0] < 10:
                print(f"Warning: {element} has limited energy range ({sorted_energies[0]:.2f} - {sorted_energies[-1]:.2f} keV)")
            
            return True
            
        except Exception as e:
            print(f"Error validating {element}: {e}")
            if element in self.data_cache:
                del self.data_cache[element]
            return False
    
    def _load_nff_file(self, element: str) -> Optional[Dict]:
        """
        Load atomic scattering factor data from a .nff file.

        Args:
            element: Element symbol (e.g., 'Fe', 'Cu')

        Returns:
            Dictionary with energy, f1, f2, and mu_rho data
        """
        # Normalize element
        element = self._normalize_element(element)

        # Try different case combinations for filename
        possible_files = [
            self.data_dir / f"{element.lower()}.nff",
            self.data_dir / f"{element.upper()}.nff",
            self.data_dir / f"{element}.nff",
            self.data_dir / f"{element.capitalize()}.nff"
        ]

        nff_file = None
        for pf in possible_files:
            if pf.exists():
                nff_file = pf
                break
            
        if nff_file is None:
            return None

        try:
            energies = []
            f1_values = []
            f2_values = []

            with open(nff_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#') or line.startswith(';'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            energy = float(parts[0])  # eV
                            f1 = float(parts[1])
                            f2 = float(parts[2])

                            # Validate values
                            if not (np.isfinite(energy) and np.isfinite(f1) and np.isfinite(f2)):
                                continue
                            
                            if energy <= 0:  # Energy must be positive
                                continue
                            
                            energies.append(energy / 1000.0)  # Convert to keV
                            f1_values.append(f1)
                            f2_values.append(f2)

                        except (ValueError, OverflowError) as e:
                            # Skip lines that can't be parsed
                            continue
                        
            if not energies or len(energies) < 4:
                print(f"Warning: Insufficient valid data in {nff_file} ({len(energies)} points)")
                return None

            # Convert to numpy arrays
            energies = np.array(energies)
            f1_values = np.array(f1_values)
            f2_values = np.array(f2_values)

            # Calculate mass attenuation coefficient from f2
            # μ/ρ = 2 r_e λ N_A f2 / A
            # Handle potential issues with wavelength calculation

            try:
                wavelengths = 12.398 / energies  # Å

                r_e = 2.818e-13  # cm
                N_A = 6.022e23
                A = self.ATOMIC_WEIGHTS.get(element, 1.0)

                # μ/ρ in cm²/g
                mu_rho = 2 * r_e * (wavelengths * 1e-8) * N_A * f2_values / A

                # Remove any invalid mu_rho values
                valid_mask = np.isfinite(mu_rho) & (mu_rho > 0)

                if not np.any(valid_mask):
                    print(f"Warning: No valid mu_rho values for {element}")
                    return None

                # Only keep valid data points
                energies = energies[valid_mask]
                f1_values = f1_values[valid_mask]
                f2_values = f2_values[valid_mask]
                mu_rho = mu_rho[valid_mask]

                return {
                    'energy': energies,      # keV
                    'f1': f1_values,
                    'f2': f2_values,
                    'mu_rho': mu_rho        # cm²/g
                }

            except Exception as e:
                print(f"Error calculating mu_rho for {element}: {e}")
                return None

        except Exception as e:
            print(f"Error reading {nff_file}: {e}")
            return None
    
    def _load_all_nff_files(self):
        """Load all available .nff files."""
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return

        loaded_count = 0
        failed_count = 0

        for element_symbol in self.ELEMENTS.values():
            data = self._load_nff_file(element_symbol)
            if data is not None:
                # Store with normalized element symbol
                normalized = self._normalize_element(element_symbol)
                self.data_cache[normalized] = data

                # Validate the data
                if self._validate_and_clean_data(normalized):
                    loaded_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1

        print(f"Loaded {loaded_count} elements successfully, {failed_count} failed or missing")
    
    def _create_interpolator(self, element: str):
        """Create interpolation function for an element's mass attenuation coefficient."""
        element = self._normalize_element(element)

        if element not in self.data_cache:
            return None

        data = self.data_cache[element]
        energies = data['energy']
        mu_rho = data['mu_rho']

        try:
            # Filter out invalid values
            valid_mask = (energies > 0) & (mu_rho > 0) & np.isfinite(energies) & np.isfinite(mu_rho)

            if not np.any(valid_mask):
                print(f"Warning: No valid data for interpolation of {element}")
                return None

            valid_energies = energies[valid_mask]
            valid_mu_rho = mu_rho[valid_mask]

            # Sort by energy
            sort_idx = np.argsort(valid_energies)
            valid_energies = valid_energies[sort_idx]
            valid_mu_rho = valid_mu_rho[sort_idx]

            # Remove duplicate energies
            unique_mask = np.concatenate([[True], np.diff(valid_energies) > 1e-10])
            valid_energies = valid_energies[unique_mask]
            valid_mu_rho = valid_mu_rho[unique_mask]

            # Check if we have enough points
            if len(valid_energies) < 4:
                print(f"Warning: Insufficient data points for {element} ({len(valid_energies)} points)")
                return None

            # Detect absorption edges (large jumps in mu_rho)
            # Calculate relative changes in mu_rho
            if len(valid_mu_rho) > 1:
                relative_changes = np.abs(np.diff(np.log10(valid_mu_rho)))

                # An edge is where there's a large sudden change
                # Use a threshold - edges typically show >20% jump in log space
                edge_threshold = 0.1  # log10 units (about 25% change)
                edge_mask = relative_changes > edge_threshold

                # Mark points near edges
                edge_indices = np.where(edge_mask)[0]

                if len(edge_indices) > 0:
                    print(f"Detected {len(edge_indices)} absorption edge(s) for {element}")

            # Use log-log space for interpolation
            log_energies = np.log10(valid_energies)
            log_mu_rho = np.log10(valid_mu_rho)

            # Check for valid log values
            if not (np.all(np.isfinite(log_energies)) and np.all(np.isfinite(log_mu_rho))):
                print(f"Warning: Invalid log values for {element}")
                return None

            # Use linear interpolation instead of cubic to avoid edge artifacts
            # Linear is more stable near discontinuities
            try:
                interpolator = interp1d(
                    log_energies,
                    log_mu_rho,
                    kind='linear',  # Changed from 'cubic' to 'linear'
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                # Test the interpolator
                _ = interpolator(log_energies[len(log_energies)//2])

            except Exception as e:
                print(f"Error: Linear interpolation failed for {element}: {e}")
                return None

            return interpolator

        except Exception as e:
            print(f"Error creating interpolator for {element}: {e}")
            return None
    
    def _load_cache(self):
        """Load cached interpolators if available."""
        cache_file = self.cache_dir / 'interpolators.pkl'
        if cache_file.exists():
            try:
                print("Loading cached interpolators...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Rebuild with normalized keys
                    for elem, interp in cached_data.items():
                        normalized = self._normalize_element(elem)
                        self.interpolators[normalized] = interp
            except Exception:
                self._create_all_interpolators()
        else:
            self._create_all_interpolators()
    
    def _create_all_interpolators(self):
        """Create interpolators for all loaded elements."""
        for element in self.data_cache.keys():
            normalized = self._normalize_element(element)
            interpolator = self._create_interpolator(normalized)
            if interpolator is not None:
                self.interpolators[normalized] = interpolator
        
        # Save to cache
        self._save_cache()
    
    def _save_cache(self):
        """Save interpolators to cache."""
        cache_file = self.cache_dir / 'interpolators.pkl'
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.interpolators, f)
        except Exception:
            pass
    
    def get_mu_rho(self, element: str, energy_kev: float) -> float:
        """
        Get mass attenuation coefficient for an element at a specific energy.

        Args:
            element: Element symbol (e.g., 'Fe', 'Cu', 'fe', 'FE')
            energy_kev: X-ray energy in keV

        Returns:
            Mass attenuation coefficient in cm²/g
        """
        element = self._normalize_element(element)

        if element not in self.interpolators:
            available = ', '.join(sorted(self.interpolators.keys()))
            raise ValueError(f"No data available for element '{element}'. Available: {available}")

        if energy_kev <= 0:
            raise ValueError(f"Energy must be positive, got {energy_kev}")

        # Check if energy is within reasonable bounds
        if element in self.data_cache:
            e_min, e_max = self.get_energy_range(element)
            if energy_kev < e_min * 0.5 or energy_kev > e_max * 2.0:
                import warnings
                warnings.warn(
                    f"Energy {energy_kev:.2f} keV is outside the recommended range "
                    f"for {element} ({e_min:.2f} - {e_max:.2f} keV). "
                    f"Extrapolation may be inaccurate.",
                    UserWarning
                )

            # Check if we're near an absorption edge
            energies = self.data_cache[element]['energy']
            mu_rho_data = self.data_cache[element]['mu_rho']

            # Find nearest data points
            idx = np.searchsorted(energies, energy_kev)
            if idx > 0 and idx < len(energies):
                # Check for large jump in mu_rho around this energy
                before_mu = mu_rho_data[idx-1]
                after_mu = mu_rho_data[idx]

                if before_mu > 0 and after_mu > 0:
                    ratio = max(before_mu, after_mu) / min(before_mu, after_mu)
                    if ratio > 2.0:  # More than 2x change suggests an edge
                        import warnings
                        warnings.warn(
                            f"Energy {energy_kev:.2f} keV is near an absorption edge for {element}. "
                            f"Results may have reduced accuracy. Consider using a different energy.",
                            UserWarning
                        )

        try:
            # Use log-log interpolation
            log_energy = np.log10(energy_kev)
            log_mu_rho = self.interpolators[element](log_energy)
            mu_rho = 10 ** log_mu_rho

            # Sanity check on result
            if not np.isfinite(mu_rho) or mu_rho <= 0:
                raise ValueError(f"Invalid interpolation result for {element} at {energy_kev} keV")

            return mu_rho

        except Exception as e:
            raise ValueError(f"Error interpolating data for {element} at {energy_kev} keV: {str(e)}")
    
    def get_available_elements(self) -> list:
        """Get list of elements with available data, sorted by atomic number."""
        # Create reverse lookup: element symbol -> atomic number
        symbol_to_z = {symbol: z for z, symbol in self.ELEMENTS.items()}

        # Get available elements and sort by atomic number
        available = list(self.data_cache.keys())
        available_sorted = sorted(available, key=lambda elem: symbol_to_z.get(elem, 999))

        return available_sorted
    
    def get_energy_range(self, element: str) -> Tuple[float, float]:
        """Get the available energy range for an element."""
        element = self._normalize_element(element)
        
        if element not in self.data_cache:
            return (0, 0)
        
        energies = self.data_cache[element]['energy']
        return (energies.min(), energies.max())


class XRDDilutionCalculator:
    """
    Calculator for determining powder sample dilution for X-ray diffraction.
    Uses atomic scattering factor data for all elements.
    """
    
    COMMON_SOURCES = {
        'Cu Kα': 8.04,
        'Mo Kα': 17.44,
        'Ag Kα': 30.0,
        'Cr Kα': 5.41,
        'Co Kα': 6.93,
    }
    
    COMMON_DILUENTS = {
        'Boron Nitride (BN)': ({'B': 0.4349, 'N': 0.5651}, 2.27),
        'Silicon (Si)': ({'Si': 1.0}, 2.33),
        'Diamond (C)': ({'C': 1.0}, 3.51),
        'Alumina (Al₂O₃)': ({'Al': 0.5293, 'O': 0.4707}, 3.95),
        'Silica (SiO₂)': ({'Si': 0.4674, 'O': 0.5326}, 2.65),
    }
    
    COMMON_SAMPLES = {
        'Iron(III) oxide (Fe₂O₃)': ({'Fe': 0.6994, 'O': 0.3006}, 5.24),
        'Calcium carbonate (CaCO₃)': ({'Ca': 0.4004, 'C': 0.1201, 'O': 0.4795}, 2.71),
        'Sodium chloride (NaCl)': ({'Na': 0.3934, 'Cl': 0.6066}, 2.16),
        'Naphthalene (C₁₀H₈)': ({'C': 0.9375, 'H': 0.0625}, 1.14),
        'Titanium dioxide (TiO₂)': ({'Ti': 0.5995, 'O': 0.4005}, 4.23),
        'Quartz (SiO₂)': ({'Si': 0.4674, 'O': 0.5326}, 2.65),
    }
    
    def __init__(self, atomic_data_manager: AtomicDataManager):
        """
        Initialize the calculator with atomic data manager.
        
        Args:
            atomic_data_manager: Instance of AtomicDataManager
        """
        self.atomic_data = atomic_data_manager
    
    def parse_composition(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Normalize composition to weight fractions."""
        total = sum(composition.values())
        if total == 0:
            raise ValueError("Total composition cannot be zero")
        return {elem: frac/total for elem, frac in composition.items()}
    
    def get_mass_attenuation(self, composition: Dict[str, float], energy_kev: float) -> float:
        """
        Calculate mass attenuation coefficient for a mixture.
        
        Args:
            composition: Dictionary of element: weight_fraction pairs
            energy_kev: X-ray energy in keV
        
        Returns:
            Mass attenuation coefficient in cm²/g
        """
        mu_rho_total = 0.0
        
        for element, weight_frac in composition.items():
            try:
                mu_rho_element = self.atomic_data.get_mu_rho(element, energy_kev)
                mu_rho_total += weight_frac * mu_rho_element
            except ValueError as e:
                raise ValueError(f"Error getting data for {element}: {str(e)}")
        
        return mu_rho_total
    
    def calculate_dilution(self, 
                          target_atten_length: float,
                          xray_energy: float,
                          sample_composition: Dict[str, float],
                          sample_density: float,
                          diluent_composition: Dict[str, float],
                          diluent_density: float,
                          packing_fraction: float = 0.6) -> Dict:
        """
        Calculate the required dilution ratio to achieve target attenuation length.
        
        Args:
            target_atten_length: Target 1/e attenuation length in mm
            xray_energy: X-ray energy in keV
            sample_composition: Dict of element: weight_fraction for sample
            sample_density: Sample density in g/cm³
            diluent_composition: Dict of element: weight_fraction for diluent
            diluent_density: Diluent density in g/cm³
            packing_fraction: Fraction of theoretical density when packed
            
        Returns:
            Dictionary with dilution parameters
        """
        # Normalize compositions
        sample_comp = self.parse_composition(sample_composition)
        diluent_comp = self.parse_composition(diluent_composition)
        
        # Convert target attenuation length to cm
        target_atten_length_cm = target_atten_length / 10.0
        
        # Calculate mass attenuation coefficients
        mu_rho_sample = self.get_mass_attenuation(sample_comp, xray_energy)
        mu_rho_diluent = self.get_mass_attenuation(diluent_comp, xray_energy)
        
        # Target linear attenuation coefficient
        target_mu = 1.0 / target_atten_length_cm  # cm⁻¹
        
        def mixture_mu(x):
            """Calculate linear attenuation for mixture with fraction x of sample"""
            mu_rho_mix = x * mu_rho_sample + (1 - x) * mu_rho_diluent
            rho_mix = packing_fraction * (x * sample_density + (1 - x) * diluent_density)
            return mu_rho_mix * rho_mix
        
        # Binary search for the right fraction
        left, right = 0.0, 1.0
        tolerance = 1e-6
        
        while right - left > tolerance:
            mid = (left + right) / 2
            mu_mid = mixture_mu(mid)
            
            if mu_mid < target_mu:
                left = mid
            else:
                right = mid
        
        sample_weight_fraction = (left + right) / 2
        diluent_weight_fraction = 1 - sample_weight_fraction
        
        # Calculate final mixture properties
        final_mu_rho = sample_weight_fraction * mu_rho_sample + diluent_weight_fraction * mu_rho_diluent
        final_density = packing_fraction * (sample_weight_fraction * sample_density + 
                                           diluent_weight_fraction * diluent_density)
        final_mu = final_mu_rho * final_density
        final_atten_length = 1.0 / final_mu  # cm
        
        # Calculate practical mixing ratios
        sample_mass_100mg = sample_weight_fraction * 100
        diluent_mass_100mg = diluent_weight_fraction * 100
        
        # Dilution ratio (mass diluent : mass sample)
        if sample_weight_fraction > 0:
            dilution_ratio = diluent_weight_fraction / sample_weight_fraction
        else:
            dilution_ratio = float('inf')
        
        return {
            'sample_weight_fraction': sample_weight_fraction,
            'diluent_weight_fraction': diluent_weight_fraction,
            'dilution_ratio': dilution_ratio,
            'sample_mass_per_100mg': sample_mass_100mg,
            'diluent_mass_per_100mg': diluent_mass_100mg,
            'mixture_density': final_density,
            'mixture_mu': final_mu,
            'achieved_attenuation_length_mm': final_atten_length * 10,
            'sample_mu_rho': mu_rho_sample,
            'diluent_mu_rho': mu_rho_diluent,
            'xray_energy_keV': xray_energy,
        }


def parse_composition_string(comp_str: str) -> Dict[str, float]:
    """Parse composition string like 'Fe:0.7, O:0.3' into dictionary."""
    composition = {}
    parts = comp_str.split(',')
    for part in parts:
        if ':' in part:
            elem, frac = part.split(':')
            composition[elem.strip()] = float(frac.strip())
    return composition

def main():
    st.set_page_config(
        page_title="SSRL XRD Sample Dilution Calculator",
        page_icon="ssrl_logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize atomic data manager (cached)
    @st.cache_resource
    def load_atomic_data():
        return AtomicDataManager(data_dir='./ScatteringFactors')
    
    with st.spinner('Loading atomic scattering factor data...'):
        atomic_data_mgr = load_atomic_data()
    
    # Initialize calculator
    calc = XRDDilutionCalculator(atomic_data_mgr)
    
    # Get available elements
    available_elements = atomic_data_mgr.get_available_elements()
    
    # Initialize widget counter for forcing updates
    if 'widget_key_counter' not in st.session_state:
        st.session_state.widget_key_counter = 0
    
    # Handle preset button clicks BEFORE rendering any widgets
    if 'preset' in st.session_state:
        if st.session_state.preset == 'fe2o3_bn':
            st.session_state.xray_source = 'Cu Kα'
            st.session_state.sample_preset = 'Iron(III) oxide (Fe₂O₃)'
            st.session_state.diluent_preset = 'Boron Nitride (BN)'
            st.session_state.target_length = 1.0
            st.session_state.sample_mode = 'Preset'
            st.session_state.diluent_mode = 'Preset'
        elif st.session_state.preset == 'caco3_si':
            st.session_state.xray_source = 'Mo Kα'
            st.session_state.sample_preset = 'Calcium carbonate (CaCO₃)'
            st.session_state.diluent_preset = 'Silicon (Si)'
            st.session_state.target_length = 1.5
            st.session_state.sample_mode = 'Preset'
            st.session_state.diluent_mode = 'Preset'
        elif st.session_state.preset == 'organic_c':
            st.session_state.xray_source = 'Cu Kα'
            st.session_state.sample_preset = 'Naphthalene (C₁₀H₈)'
            st.session_state.diluent_preset = 'Diamond (C)'
            st.session_state.target_length = 2.0
            st.session_state.sample_mode = 'Preset'
            st.session_state.diluent_mode = 'Preset'
        
        # Increment counter to force widget recreation
        st.session_state.widget_key_counter += 1
        del st.session_state.preset
    
    # Create unique widget key suffix
    wk = st.session_state.widget_key_counter
    
    # Header
    st.markdown('<h1 class="main-header">🔬 XRD Sample Dilution Calculator</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Enhanced with CXRO data for {len(available_elements)} elements | '
                f'Calculate optimal powder dilution for X-ray diffraction</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Quick Start")
        
        st.subheader("Presets")
        if st.button("Fe₂O₃ in BN (Cu Kα)", width='stretch'):
            st.session_state.preset = 'fe2o3_bn'
            st.rerun()
        if st.button("CaCO₃ in Si (Mo Kα)", width='stretch'):
            st.session_state.preset = 'caco3_si'
            st.rerun()
        if st.button("Organic in Diamond", width='stretch'):
            st.session_state.preset = 'organic_c'
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("ℹ️ Data Info")
        st.info(f"""
        **Loaded Elements:** {len(available_elements)}
        
        **Energy Range:** 0.1 - 100 keV
        
        **Data Source:** CXRO (Lawrence Berkeley Lab)
        """)
        
        st.markdown("---")
        
        st.subheader("🔍 Energy Range Checker")
        if available_elements:
            # Create reverse lookup for atomic numbers
            symbol_to_z = {symbol: z for z, symbol in atomic_data_mgr.ELEMENTS.items()}
            
            # Create display labels with atomic numbers
            element_labels = [f"{symbol_to_z.get(elem, '?')}. {elem}" for elem in available_elements]
            
            selected_index = st.selectbox(
                "Select element:",
                range(len(available_elements)),
                format_func=lambda i: element_labels[i],
                key=f'energy_check_element_{wk}'
            )
            
            selected_element = available_elements[selected_index]
            e_min, e_max = atomic_data_mgr.get_energy_range(selected_element)
            st.success(f"**{element_labels[selected_index]}:** {e_min:.2f} - {e_max:.2f} keV")
        else:
            st.warning("No element data loaded")
        
        st.markdown("---")
        
        with st.expander("📖 Available Elements"):
            if available_elements:
                # Display in columns
                cols_per_row = 5
                for i in range(0, len(available_elements), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(available_elements):
                            col.write(available_elements[i + j])
            else:
                st.warning("No .nff files found. Please add element data files.")
    
    # Check for no data
    if not available_elements:
        st.error("""
        ⚠️ **No atomic scattering factor data loaded!**
        
        Please ensure .nff files are in the same directory as this script.
        
        Expected format: `element.nff` (e.g., `fe.nff`, `cu.nff`)
        
        You can download .nff files from: http://henke.lbl.gov/optical_constants/sf/
        """)
        st.stop()
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("⚙️ Experimental Parameters")
        
        # X-ray source
        xray_mode = st.radio(
            "X-ray Source:",
            ["Common Source", "Custom Energy"],
            horizontal=True,
            key=f'xray_mode_{wk}'
        )
        
        if xray_mode == "Common Source":
            # Get the default or preset xray source
            default_source = st.session_state.get('xray_source', 'Cu Kα')
            source_list = list(calc.COMMON_SOURCES.keys())
            source_index = source_list.index(default_source) if default_source in source_list else 0
            
            xray_source = st.selectbox(
                "Select source:",
                source_list,
                index=source_index,
                key=f'xray_source_widget_{wk}'
            )
            xray_energy = calc.COMMON_SOURCES[xray_source]
            st.info(f"Energy: {xray_energy} keV")
        else:
            xray_energy = st.number_input(
                "X-ray energy (keV):",
                min_value=0.1,
                max_value=100.0,
                value=8.04,
                step=0.1,
                key=f'custom_energy_{wk}'
            )
        
        # Target parameters
        st.subheader("Target Parameters")
        target_atten_length = st.number_input(
            "Target attenuation length (mm):",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.get('target_length', 1.0),
            step=0.1,
            key=f'target_length_input_{wk}'
        )
        
        packing_fraction = st.slider(
            "Packing fraction:",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Typical values: 0.5-0.7 for powders",
            key=f'packing_fraction_{wk}'
        )
    
    with col2:
        st.header("🧪 Sample Composition")
        
        # Sample input
        # Get default mode from session state
        default_sample_mode = st.session_state.get('sample_mode', 'Preset')
        
        sample_mode = st.radio(
            "Sample input method:",
            ["Preset", "Custom"],
            horizontal=True,
            index=0 if default_sample_mode == 'Preset' else 1,
            key=f'sample_mode_widget_{wk}'
        )
        
        if sample_mode == "Preset":
            # Get default sample from session state
            default_sample = st.session_state.get('sample_preset', list(calc.COMMON_SAMPLES.keys())[0])
            sample_list = list(calc.COMMON_SAMPLES.keys())
            sample_index = sample_list.index(default_sample) if default_sample in sample_list else 0
            
            sample_name = st.selectbox(
                "Select sample:",
                sample_list,
                index=sample_index,
                key=f'sample_preset_widget_{wk}'
            )
            
            sample_comp, sample_density_default = calc.COMMON_SAMPLES[sample_name]
            
            # Display composition - DYNAMIC KEY!
            comp_str = ", ".join([f"{k}:{v:.4f}" for k, v in sample_comp.items()])
            st.text_input(
                "Composition:", 
                value=comp_str, 
                disabled=True, 
                key=f'sample_comp_display_{sample_name}_{wk}'
            )
            
            # Dynamic key - updates when preset changes!
            sample_density = st.number_input(
                "Sample density (g/cm³):",
                value=float(sample_density_default),
                step=0.01,
                format="%.3f",
                key=f'sample_density_{sample_name}_{wk}'
            )
            
        else:
            sample_name = st.text_input("Sample name (optional):", value="Custom Sample", key=f'sample_name_custom_{wk}')
            sample_comp_str = st.text_input(
                "Composition (Element:Fraction):",
                value="Fe:0.7, O:0.3",
                help="Example: Fe:0.7, O:0.3",
                key=f'sample_comp_str_{wk}'
            )
            try:
                sample_comp = parse_composition_string(sample_comp_str)
            except:
                st.error("Invalid composition format. Use: Element:Fraction, Element:Fraction")
                sample_comp = {'Fe': 0.7, 'O': 0.3}
            
            sample_density = st.number_input(
                "Sample density (g/cm³):",
                min_value=0.1,
                max_value=25.0,
                value=5.0,
                step=0.1,
                key=f'sample_density_custom_{wk}'
            )
    
    # Diluent section
    st.header("🧊 Diluent Composition")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Get default mode from session state
        default_diluent_mode = st.session_state.get('diluent_mode', 'Preset')
        
        diluent_mode = st.radio(
            "Diluent input method:",
            ["Preset", "Custom"],
            horizontal=True,
            index=0 if default_diluent_mode == 'Preset' else 1,
            key=f'diluent_mode_widget_{wk}'
        )
        
        if diluent_mode == "Preset":
            # Get default diluent from session state
            default_diluent = st.session_state.get('diluent_preset', list(calc.COMMON_DILUENTS.keys())[0])
            diluent_list = list(calc.COMMON_DILUENTS.keys())
            diluent_index = diluent_list.index(default_diluent) if default_diluent in diluent_list else 0
            
            diluent_name = st.selectbox(
                "Select diluent:",
                diluent_list,
                index=diluent_index,
                key=f'diluent_preset_widget_{wk}'
            )
            
            diluent_comp, diluent_density_default = calc.COMMON_DILUENTS[diluent_name]
            
            # Display composition - DYNAMIC KEY!
            comp_str = ", ".join([f"{k}:{v:.4f}" for k, v in diluent_comp.items()])
            st.text_input(
                "Composition:", 
                value=comp_str, 
                disabled=True, 
                key=f'diluent_comp_display_{diluent_name}_{wk}'
            )
            
        else:
            diluent_name = st.text_input("Diluent name (optional):", value="Custom Diluent", key=f'diluent_name_custom_{wk}')
            diluent_comp_str = st.text_input(
                "Composition (Element:Fraction):",
                value="B:0.43, N:0.57",
                help="Example: Si:1.0 or B:0.43, N:0.57",
                key=f'diluent_comp_str_{wk}'
            )
            try:
                diluent_comp = parse_composition_string(diluent_comp_str)
            except:
                st.error("Invalid composition format")
                diluent_comp = {'Si': 1.0}
    
    with col4:
        if diluent_mode == "Preset":
            # Dynamic key - updates when preset changes!
            diluent_density = st.number_input(
                "Diluent density (g/cm³):",
                min_value=0.1,
                max_value=25.0,
                value=float(diluent_density_default),
                step=0.01,
                format="%.3f",
                key=f'diluent_density_{diluent_name}_{wk}'
            )
        else:
            diluent_density = st.number_input(
                "Diluent density (g/cm³):",
                min_value=0.1,
                max_value=25.0,
                value=2.3,
                step=0.01,
                format="%.3f",
                key=f'diluent_density_custom_{wk}'
            )
    
    # Calculate button
    st.markdown("---")
    if st.button("🧮 Calculate Dilution", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating..."):
                # Capture warnings
                import warnings
                warning_messages = []
                
                def warning_handler(message, category, filename, lineno, file=None, line=None):
                    warning_messages.append(str(message))
                
                old_warning = warnings.showwarning
                warnings.showwarning = warning_handler
                
                # Validate elements
                all_elements = list(sample_comp.keys()) + list(diluent_comp.keys())
                invalid_elements = [e for e in all_elements if atomic_data_mgr._normalize_element(e) not in available_elements]
                
                if invalid_elements:
                    st.error(f"❌ No data available for elements: {', '.join(invalid_elements)}")
                    st.info(f"Available elements: {', '.join(available_elements)}")
                    st.stop()
                
                results = calc.calculate_dilution(
                    target_atten_length=target_atten_length,
                    xray_energy=xray_energy,
                    sample_composition=sample_comp,
                    sample_density=sample_density,
                    diluent_composition=diluent_comp,
                    diluent_density=diluent_density,
                    packing_fraction=packing_fraction
                )
                
                # Restore warning handler
                warnings.showwarning = old_warning
                
                # Display any warnings
                if warning_messages:
                    for msg in warning_messages:
                        st.warning(f"⚠️ {msg}")
            
            # Display results
            st.success("✅ Calculation Complete!")
            
            st.header("📊 Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sample Fraction</div>
                    <div class="metric-value">{results['sample_weight_fraction']*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Diluent Fraction</div>
                    <div class="metric-value">{results['diluent_weight_fraction']*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Dilution Ratio</div>
                    <div class="metric-value">{results['dilution_ratio']:.3f}</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">g diluent / g sample</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Achieved Length</div>
                    <div class="metric-value">{results['achieved_attenuation_length_mm']:.3f}</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">mm</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed results
            st.subheader("📝 Detailed Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sample Properties:**")
                st.write(f"- Mass attenuation coefficient: {results['sample_mu_rho']:.3f} cm²/g")
                st.write(f"- Density: {sample_density:.3f} g/cm³")
                st.write(f"- X-ray energy: {results['xray_energy_keV']:.2f} keV")
                
                st.markdown("**Diluent Properties:**")
                st.write(f"- Mass attenuation coefficient: {results['diluent_mu_rho']:.3f} cm²/g")
                st.write(f"- Density: {diluent_density:.3f} g/cm³")
            
            with col2:
                st.markdown("**Mixture Properties:**")
                st.write(f"- Packed density: {results['mixture_density']:.3f} g/cm³")
                st.write(f"- Linear attenuation: {results['mixture_mu']:.4f} cm⁻¹")
                st.write(f"- Attenuation length: {results['achieved_attenuation_length_mm']:.3f} mm")
                st.write(f"- Packing fraction: {packing_fraction:.2f}")
            
            # Mixing recipe
            st.subheader("🥄 Mixing Recipe")
            
            # Create table with different total masses
            total_masses = [50, 100, 200, 500, 1000]
            recipe_data = []
            
            for total_mass in total_masses:
                sample_mass = results['sample_weight_fraction'] * total_mass
                diluent_mass = results['diluent_weight_fraction'] * total_mass
                recipe_data.append({
                    'Total Mass (mg)': total_mass,
                    'Sample Mass (mg)': f"{sample_mass:.2f}",
                    'Diluent Mass (mg)': f"{diluent_mass:.2f}"
                })
            
            recipe_df = pd.DataFrame(recipe_data)
            st.dataframe(recipe_df, width='stretch', hide_index=True)
            
            # Additional recommendations
            st.subheader("💡 Recommendations")
            
            if results['sample_weight_fraction'] < 0.05:
                st.warning(
                    "⚠️ Sample fraction is very low (<5%). "
                    "Ensure thorough mixing for homogeneity."
                )
            
            if results['sample_weight_fraction'] > 0.95:
                st.info(
                    "ℹ️ Sample fraction is very high (>95%). "
                    "Dilution may not be necessary."
                )
            
            optimal_thickness = results['achieved_attenuation_length_mm'] * 2.5
            st.info(
                f"💡 Recommended sample thickness: "
                f"{optimal_thickness:.2f} mm (2.5× attenuation length)"
            )
            
            # Download results
            st.subheader("💾 Export Results")
            
            results_text = f"""
XRD Sample Dilution Calculation Results
{'='*60}

Sample: {sample_name if sample_mode == 'Preset' else 'Custom Sample'}
Diluent: {diluent_name if diluent_mode == 'Preset' else 'Custom Diluent'}
X-ray Energy: {xray_energy} keV
Target Attenuation Length: {target_atten_length} mm
Packing Fraction: {packing_fraction}

Sample Composition: {', '.join([f"{k}:{v:.4f}" for k, v in sample_comp.items()])}
Sample Density: {sample_density} g/cm³

Diluent Composition: {', '.join([f"{k}:{v:.4f}" for k, v in diluent_comp.items()])}
Diluent Density: {diluent_density} g/cm³

{'='*60}
RESULTS
{'='*60}

Sample weight fraction: {results['sample_weight_fraction']*100:.2f}%
Diluent weight fraction: {results['diluent_weight_fraction']*100:.2f}%
Dilution ratio: {results['dilution_ratio']:.3f} (g diluent / g sample)

Mixture Properties:
- Packed density: {results['mixture_density']:.3f} g/cm³
- Linear attenuation: {results['mixture_mu']:.4f} cm⁻¹
- Achieved attenuation length: {results['achieved_attenuation_length_mm']:.3f} mm

Sample Properties:
- Mass attenuation coefficient: {results['sample_mu_rho']:.3f} cm²/g
- Density: {sample_density:.3f} g/cm³

Diluent Properties:
- Mass attenuation coefficient: {results['diluent_mu_rho']:.3f} cm²/g
- Density: {diluent_density:.3f} g/cm³

{'='*60}
MIXING RECIPE
{'='*60}

For 50 mg total:
- Sample: {results['sample_weight_fraction']*50:.2f} mg
- Diluent: {results['diluent_weight_fraction']*50:.2f} mg

For 100 mg total:
- Sample: {results['sample_mass_per_100mg']:.2f} mg
- Diluent: {results['diluent_mass_per_100mg']:.2f} mg

For 200 mg total:
- Sample: {results['sample_weight_fraction']*200:.2f} mg
- Diluent: {results['diluent_weight_fraction']*200:.2f} mg

For 500 mg total:
- Sample: {results['sample_weight_fraction']*500:.2f} mg
- Diluent: {results['diluent_weight_fraction']*500:.2f} mg

For 1000 mg total:
- Sample: {results['sample_weight_fraction']*1000:.2f} mg
- Diluent: {results['diluent_weight_fraction']*1000:.2f} mg

Recommended sample thickness: {optimal_thickness:.2f} mm

{'='*60}
Data Source: CXRO atomic scattering factors (Lawrence Berkeley Lab)
Calculation performed with {len(available_elements)} elements available
{'='*60}
"""
            
            st.download_button(
                label="📄 Download Results as Text File",
                data=results_text,
                file_name="xrd_dilution_results.txt",
                mime="text/plain",
                width='stretch'
            )
            
        except Exception as e:
            st.error(f"❌ Error during calculation: {str(e)}")
            with st.expander("Show error details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Footer with help
    st.markdown("---")
    
    with st.expander("📖 Help & Information"):
        st.markdown("""
        ### How to Use:
        
        1. **Select X-ray source** or enter custom energy (0.1-100 keV)
        2. **Enter target attenuation length** (typically 0.5-2 mm)
        3. **Define your sample** composition and density
        4. **Choose diluent** material
        5. Click **"Calculate Dilution"** to get results
        
        ### Composition Format:
        
        - Enter as `Element:Fraction` pairs
        - Separate with commas
        - Example: `Fe:0.7, O:0.3` or `C:1.0`
        - Fractions will be automatically normalized
        
        ### About Attenuation Length:
        
        The **attenuation length** (1/e length) is the depth at which X-ray intensity 
        drops to approximately 37% of its initial value.
        
        **Typical values:**
        - **0.5-1.0 mm:** Highly absorbing samples (heavy elements, high density)
        - **1.0-2.0 mm:** Moderate absorption (most common samples)
        - **>2.0 mm:** Weakly absorbing samples (light elements, low density)
        
        **Optimal sample thickness** is usually 2-3× the attenuation length for good 
        statistics while minimizing absorption effects.
        
        ### Packing Fraction:
        
        The **packing fraction** represents the ratio of the actual packed density to 
        the theoretical crystal density:
        
        - **0.5-0.6:** Loosely packed powders, fluffy materials
        - **0.6-0.7:** Typical well-mixed powders (recommended default)
        - **0.7-0.9:** Compacted or pressed samples
        
        ### Energy Range:
        
        - The calculator supports energies from 0.1 to 100 keV
        - Data accuracy depends on CXRO coverage for each element
        - Most accurate in the 1-30 keV range (typical XRD sources)
        - Use the Energy Range Checker in the sidebar to verify data availability
        
        ### Data Source:
        
        Atomic scattering factors from **CXRO** (Center for X-ray Optics, Lawrence Berkeley National Laboratory)
        
        - High-quality experimental and theoretical data
        - Covers elements H through U
        - Energy-dependent f1 and f2 values
        - Converted to mass attenuation coefficients using standard formulas
        
        ### Tips for Best Results:
        
        1. **Use presets** for common materials to get started quickly
        2. **Verify element availability** using the sidebar checker
        3. **Check energy ranges** - ensure your X-ray energy is within data range
        4. **Start with standard packing fraction** (0.6) and adjust based on your mixing method
        5. **Consider sample homogeneity** - lower fractions require more careful mixing
        6. **Download results** for your lab notebook and future reference
        
        ### Common X-ray Sources:
        
        - **Cu Kα:** 8.04 keV (most common laboratory source)
        - **Mo Kα:** 17.44 keV (for weakly absorbing samples)
        - **Ag Kα:** 30.0 keV (high energy applications)
        - **Cr Kα:** 5.41 keV (low energy, high absorption)
        - **Co Kα:** 6.93 keV (alternative to Cu)
        
        ### File Requirements:
        
        The program requires .nff files in the same directory:
        
        - **Format:** 3-column ASCII (Energy[eV], f1, f2)
        - **Naming:** element.nff (e.g., fe.nff, cu.nff)
        - **Download from:** http://henke.lbl.gov/optical_constants/sf/
        
        Example file content:
        ```
        # Energy(eV)  f1      f2
        100.0         5.234   0.123
        200.0         5.156   0.234
        ...
        ```
        
        ### Troubleshooting:
        
        - **Element not found:** Check that the .nff file exists and is readable
        - **Interpolation errors:** Some elements may have limited energy coverage
        - **Unexpected results:** Verify composition fractions sum to ~1.0
        - **Energy out of range:** Use the sidebar checker to verify element data range
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666;'>
            <p><strong>XRD Sample Dilution Calculator</strong> | 
            Enhanced with CXRO atomic scattering factors</p>
            <p>Loaded {len(available_elements)} elements | 
            Mass attenuation data from Lawrence Berkeley Lab</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()