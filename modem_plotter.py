"""
Author: Hakim Rezayee
Affiliation: Geoscience Enterprise Inc. (GSE), Tokyo.
Date: January 22, 2026
Description: Script to display the resistivity model result from ModEM with MT station plotting.

Note: Before running this module, use 'modem2xyz' to extract the data 
from your ModEM output and save it as a CSV file.

Please do not change anything if not necessary!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
import warnings
import sys
from typing import Optional, Tuple, List, Dict, Any, Union

warnings.filterwarnings('ignore')

class ModemPlotter:
    
    def __init__(self, csv_file: Optional[str] = None, config: Optional[Dict] = None):

        self.csv_file = csv_file
        self.x = None
        self.y = None
        self.z = None
        self.rho = None
        self.mt_stations = None 
        self.loaded = False
        
        self.config = {
            # DATA RANGES (in km) 
            'x_range_km': (-50.0, 50.0),   
            'y_range_km': (-60.0, 60.0),   
            'z_range_km': (0, 40.0),        
            
            'dpi': 300,
            'fontsize': 11,
            'title_fontsize': 12,
            'label_fontsize': 12,
            'tick_fontsize': 12,
            'figsize': None,                  # Auto-calculate if None
            
            'cmap': 'jet_r',
            'vmin': 0.0,
            'vmax': 5.0,
            'log_scale': True,
            'colorbar_label': 'Log$_{10}$ Resistivity (Ω m)',
            'colorbar_ticks': [0, 1, 2, 3, 4, 5],
            'data_in_natural_log': True,   
            'convert_to_log10': True,   

            'cbar_width': 0.3,     
            'cbar_height': 0.015,   
            'cbar_y_position': 0.04,   
            
            'exclude_air': True,
            'air_resistivity_threshold': 1e10,
            'depth_tolerance': 10.0,          # Meters
            'position_tolerance': 1000.0,     # Meters for vertical slices

            'output_dir': 'output',
            'save_format': 'png',
            'save_dpi': 300,
            'tight_layout': False,   
            
            'units': 'km',                    # 'km' or 'm'
             
            'wspace': 0.02,
            'hspace': 0.02, 
            
            'axis_padding': 0.0,         
            'axis_linewidth': 1.0,  

            'linewidth': 0.8,
            'grid_alpha': 0.15,
            'grid_linestyle': ':',
            'spine_width': 1.0,   
            'tick_length': 3,  
            'tick_width': 0.8,
            
            'verbose': True,
            'debug': False,
             
            'pcolormesh_edgecolor': 'face', 
            'pcolormesh_linewidth': 0.0,
            'grid_visible': True, 
            'zero_line_visible': False,      
            'interpolate_missing': True,      # Interpolate missing values to avoid gaps
            'edge_smoothing': True,           # Smooth cell edges
            
            'show_all_ticks': True,   
            'n_ticks': 5, 
            'hide_top_right_ticks': True, 
            
            'mt_station_file': None,  
            'mt_station_color': 'white', 
            'mt_station_marker': '^',         
            'mt_station_size': 25,  
            'mt_station_edgecolor': 'black', 
            'mt_station_linewidth': 0.5,     
            'mt_station_alpha': 0.9,     
            'plot_mt_stations': True, 
            'mt_units': 'km',
            
            'add_subplot_labels': True,  
            'subplot_labels': ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)'],  # Labels to use
            'subplot_label_fontsize': 14,    
            'subplot_label_weight': 'bold',   
            'subplot_label_position': (0.02, 0.98), 
            'subplot_label_color': 'black', 
            'subplot_label_bg': None,  
        }

        if config:
            self._update_config(config)

        self._convert_units()

        if csv_file:
            self.load_data(csv_file)

        if self.config['mt_station_file']:
            self.load_mt_stations(self.config['mt_station_file'])
    
    def _update_config(self, config: Dict) -> None:
        """Safely update configuration"""
        for key, value in config.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                if self.config['verbose']:
                    print(f" Unknown config parameter '{key}'")
                self.config[key] = value
    
    def _convert_units(self) -> None:
        """Convert all ranges to consistent units (meters)"""
        if self.config['units'] == 'km':
            self.config['x_range'] = (
                self.config['x_range_km'][0] * 1000,
                self.config['x_range_km'][1] * 1000
            )
            self.config['y_range'] = (
                self.config['y_range_km'][0] * 1000,
                self.config['y_range_km'][1] * 1000
            )
            self.config['z_range'] = (
                self.config['z_range_km'][0] * 1000,
                self.config['z_range_km'][1] * 1000
            )
        else:
            self.config['x_range'] = self.config['x_range_km']
            self.config['y_range'] = self.config['y_range_km']
            self.config['z_range'] = self.config['z_range_km']
    
    def _log(self, message: str, level: str = 'info') -> None:
        """Log messages based on verbosity settings"""
        if not self.config['verbose'] and level != 'error':
            return
        
        prefix = {
            'info': ' ',
            'warning': ' ',
            'error': ' ',
            'success': ' ',
            'debug': ' '
        }.get(level, ' ')
        
        if level == 'debug' and not self.config['debug']:
            return
        
        print(f"{prefix} {message}")
    
    def load_mt_stations(self, station_file: str) -> bool:
        try:
            if not Path(station_file).exists():
                self._log(f"MT station file not found: {station_file}", 'error')
                return False

            data = None

            try:
                data = np.loadtxt(station_file, skiprows=1)
            except:
                try:
                    data = np.loadtxt(station_file)
                except Exception as e:
                    self._log(f"Failed to load MT station file: {e}", 'error')
                    return False
            
            if data is None or data.ndim != 2 or data.shape[1] < 2:
                raise ValueError(f"Expected at least 2 columns, got {data.shape[1] if data.ndim == 2 else 'unknown'}")

            if self.config['mt_units'] == 'km':
                self.mt_stations = {
                    'x': data[:, 0] * 1000,
                    'y': data[:, 1] * 1000
                }
            else:
                self.mt_stations = {
                    'x': data[:, 0],
                    'y': data[:, 1]
                }
            
            self._log(f"MT stations loaded: {len(self.mt_stations['x'])} stations")
            self._log(f"Station X range: {self.mt_stations['x'].min()/1000:.1f} to {self.mt_stations['x'].max()/1000:.1f} km")
            self._log(f"Station Y range: {self.mt_stations['y'].min()/1000:.1f} to {self.mt_stations['y'].max()/1000:.1f} km")
            
            # Check if stations are within resistivity data range
            if self.loaded:
                x_min, x_max = self.x.min(), self.x.max()
                y_min, y_max = self.y.min(), self.y.max()
                
                within_x = np.sum((self.mt_stations['x'] >= x_min) & (self.mt_stations['x'] <= x_max))
                within_y = np.sum((self.mt_stations['y'] >= y_min) & (self.mt_stations['y'] <= y_max))
                
                if within_x < len(self.mt_stations['x']):
                    self._log(f"Warning: {len(self.mt_stations['x']) - within_x} stations outside X range", 'warning')
                if within_y < len(self.mt_stations['y']):
                    self._log(f"Warning: {len(self.mt_stations['y']) - within_y} stations outside Y range", 'warning')
            
            return True
            
        except Exception as e:
            self._log(f"Failed to load MT stations: {e}", 'error')
            return False
    
    def load_data(self, csv_file: str) -> 'ModemPlotter':

        try:
            if not Path(csv_file).exists():
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            
            if data.shape[1] < 4:
                raise ValueError(f"CSV must have at least 4 columns")
            
            self.x = data[:, 0]
            self.y = data[:, 1]
            self.z = data[:, 2]
            self.rho = data[:, 3]
            self.loaded = True
            
            self.is_natural_log_data = False
            if self.rho.min() < 0:

                median_val = np.median(self.rho)
                self.is_natural_log_data = median_val < -1  
            
            if self.config['verbose']:
                self._log(f"Data statistics:")
                self._log(f"  Min resistivity: {self.rho.min():.2f}")
                self._log(f"  Max resistivity: {self.rho.max():.2f}")
                self._log(f"  Median resistivity: {np.median(self.rho):.2f}")
                self._log(f"  Data appears to be in natural log: {self.is_natural_log_data}")
            
            self._log(f"Data loaded: {len(self.x):,} points")
            self._log(f"X: {self.config['x_range_km'][0]} to {self.config['x_range_km'][1]} km")
            self._log(f"Y: {self.config['y_range_km'][0]} to {self.config['y_range_km'][1]} km")
            self._log(f"Z: {self.config['z_range_km'][0]} to {self.config['z_range_km'][1]} km")
            
            return self
            
        except Exception as e:
            self._log(f"Failed to load data: {e}", 'error')
            raise
    
    def _calculate_edges(self, centers: np.ndarray) -> np.ndarray:

        if len(centers) < 2:
            return centers
        
        edges = np.zeros(len(centers) + 1)
        edges[1:-1] = (centers[1:] + centers[:-1]) / 2
        edges[0] = centers[0] - (centers[1] - centers[0]) / 2
        edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
        
        return edges
    
    def _create_regular_grid(self, x_vals: np.ndarray, y_vals: np.ndarray, 
                           z_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        try:
            x_unique = np.sort(np.unique(x_vals))
            y_unique = np.sort(np.unique(y_vals))
            
            if len(x_unique) < 2 or len(y_unique) < 2:
                self._log(f"Insufficient unique values", 'warning')
                return None, None, None
            
            x_edges = self._calculate_edges(x_unique)
            y_edges = self._calculate_edges(y_unique)
            
            X_edges, Y_edges = np.meshgrid(x_edges, y_edges)
            Z = np.full((len(y_unique), len(x_unique)), np.nan)
            
            coord_dict = {}
            for i, xi in enumerate(x_unique):
                for j, yj in enumerate(y_unique):
                    coord_dict[(xi, yj)] = (j, i)
            
            for xi, yi, zi in zip(x_vals, y_vals, z_vals):
                if (xi, yi) in coord_dict:
                    j, i = coord_dict[(xi, yi)]
                    Z[j, i] = zi
            
            if self.config['interpolate_missing'] and np.any(np.isnan(Z)):
                from scipy import interpolate

                mask = ~np.isnan(Z)
                if np.sum(mask) > 4:  
                    x_coords, y_coords = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))

                    points = np.column_stack([x_coords[mask], y_coords[mask]])
                    values = Z[mask]

                    try:
                        interp = interpolate.LinearNDInterpolator(points, values)
                        eval_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
                        Z_interp = interp(eval_points).reshape(Z.shape)
                        
                        nan_mask = np.isnan(Z)
                        Z[nan_mask] = Z_interp[nan_mask]
                    except:
                        pass  
                        
                    if np.any(np.isnan(Z)):

                        from scipy.ndimage import distance_transform_edt
                        mask = ~np.isnan(Z)
                        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
                        Z = Z[tuple(indices)]
            
            nan_ratio = np.isnan(Z).sum() / Z.size
            if nan_ratio > 0.1:  
                self._log(f"High NaN ratio after interpolation: {nan_ratio:.1%}", 'warning')
            
            return X_edges, Y_edges, Z
            
        except Exception as e:
            self._log(f"Error creating grid: {e}", 'error')
            return None, None, None
    
    def _prepare_data_for_plot(self, x_vals: np.ndarray, y_vals: np.ndarray, 
                             rho_vals: np.ndarray, x_range: Tuple[float, float], 
                             y_range: Tuple[float, float]) -> Tuple[Optional[np.ndarray], ...]:

        try:
            x_mask = (x_vals >= x_range[0]) & (x_vals <= x_range[1])
            y_mask = (y_vals >= y_range[0]) & (y_vals <= y_range[1])
            mask = x_mask & y_mask
            
            if not np.any(mask):
                self._log(f"No data in specified range", 'warning')
                return None, None, None
            
            if np.sum(mask) < 4:
                self._log(f"Insufficient data points: {np.sum(mask)}", 'warning')
                return None, None, None
            
            x_masked = x_vals[mask]
            y_masked = y_vals[mask]
            rho_masked = rho_vals[mask]
            
            x_min, x_max = x_range
            y_min, y_max = y_range
            
            if not self.config['zero_line_visible']:

                zero_x_mask = np.abs(x_masked) > 1e-6
                zero_y_mask = np.abs(y_masked) > 1e-6
                mask_nonzero = zero_x_mask & zero_y_mask
                
                if np.sum(~mask_nonzero) > 0:
                    x_masked = x_masked[mask_nonzero]
                    y_masked = y_masked[mask_nonzero]
                    rho_masked = rho_masked[mask_nonzero]
            
            if not np.any(np.abs(x_masked - x_min) < 1e-6):
                
                y_boundary = np.linspace(y_min, y_max, 5)
                x_boundary = np.full_like(y_boundary, x_min)
                rho_boundary = np.full_like(y_boundary, np.nan)
                
                x_masked = np.append(x_masked, x_boundary)
                y_masked = np.append(y_masked, y_boundary)
                rho_masked = np.append(rho_masked, rho_boundary)
            
            if not np.any(np.abs(x_masked - x_max) < 1e-6):
                y_boundary = np.linspace(y_min, y_max, 5)
                x_boundary = np.full_like(y_boundary, x_max)
                rho_boundary = np.full_like(y_boundary, np.nan)
                
                x_masked = np.append(x_masked, x_boundary)
                y_masked = np.append(y_masked, y_boundary)
                rho_masked = np.append(rho_masked, rho_boundary)
            
            if not np.any(np.abs(y_masked - y_min) < 1e-6):
                x_boundary = np.linspace(x_min, x_max, 5)
                y_boundary = np.full_like(x_boundary, y_min)
                rho_boundary = np.full_like(x_boundary, np.nan)
                
                x_masked = np.append(x_masked, x_boundary)
                y_masked = np.append(y_masked, y_boundary)
                rho_masked = np.append(rho_masked, rho_boundary)
            
            if not np.any(np.abs(y_masked - y_max) < 1e-6):
                x_boundary = np.linspace(x_min, x_max, 5)
                y_boundary = np.full_like(x_boundary, y_max)
                rho_boundary = np.full_like(x_boundary, np.nan)
                
                x_masked = np.append(x_masked, x_boundary)
                y_masked = np.append(y_masked, y_boundary)
                rho_masked = np.append(rho_masked, rho_boundary)
            
            X, Y, Z = self._create_regular_grid(x_masked, y_masked, rho_masked)
            
            if X is None:
                return None, None, None
            
            if self.config['exclude_air']:
                threshold = self.config['air_resistivity_threshold']
                if self.config['data_in_natural_log']:
                    log_threshold = np.log(threshold) if threshold > 0 else -100
                    Z = np.ma.masked_where(Z >= log_threshold, Z)
                elif self.is_natural_log_data:
                    log_threshold = np.log(threshold) if threshold > 0 else -100
                    Z = np.ma.masked_where(Z >= log_threshold, Z)
                else:
                    Z = np.ma.masked_where(Z >= threshold, Z)
            
            # Convert from natural log to log10 for plotting
            if self.config['log_scale']:
                if self.config['data_in_natural_log'] or self.is_natural_log_data:

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        Z_plot = Z / np.log(10)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        Z_plot = np.ma.log10(Z)
            else:
                if self.config['data_in_natural_log'] or self.is_natural_log_data:
                    Z_plot = np.ma.exp(Z)
                else:
                    Z_plot = Z
            
            # Apply edge smoothing to reduce artifacts
            if self.config['edge_smoothing'] and np.ma.is_masked(Z_plot):
                # Create a small Gaussian filter to smooth edges
                from scipy.ndimage import gaussian_filter
                Z_data = Z_plot.data.copy()
                mask = Z_plot.mask
                
                if np.any(mask):
                    # Create distance transform to find nearest valid values
                    from scipy.ndimage import distance_transform_edt
                    
                    # Get indices of valid data
                    valid_indices = np.where(~mask)
                    if len(valid_indices[0]) > 0:
                        # Create temporary array for smoothing
                        Z_temp = Z_data.copy()
                        
                        # Apply mild Gaussian filter to reduce sharp edges
                        Z_temp = gaussian_filter(Z_temp, sigma=0.5, mode='nearest')
                        Z_temp[~mask] = Z_data[~mask]
                        Z_plot = np.ma.array(Z_temp, mask=mask)
            
            if np.ma.is_masked(Z_plot):
                masked_ratio = Z_plot.mask.sum() / Z_plot.size
                if masked_ratio > 0.8:
                    self._log(f"High masked ratio: {masked_ratio:.1%}", 'warning')
            
            return X, Y, Z_plot
            
        except Exception as e:
            self._log(f"Error preparing data: {e}", 'error')
            return None, None, None
    
    def _add_mt_stations(self, ax: plt.Axes) -> None:

        if self.mt_stations is None or not self.loaded:
            return
        
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            mask_x = (self.mt_stations['x'] >= xlim[0]) & (self.mt_stations['x'] <= xlim[1])
            mask_y = (self.mt_stations['y'] >= ylim[0]) & (self.mt_stations['y'] <= ylim[1])
            mask = mask_x & mask_y
            
            if np.sum(mask) == 0:
                return
            
            # Plot MT stations
            ax.scatter(
                self.mt_stations['x'][mask],
                self.mt_stations['y'][mask],
                marker=self.config['mt_station_marker'],
                s=self.config['mt_station_size'],
                c=self.config['mt_station_color'],
                edgecolors=self.config['mt_station_edgecolor'],
                linewidth=self.config['mt_station_linewidth'],
                alpha=self.config['mt_station_alpha'],
                zorder=10,  # Ensure stations are on top
                label='MT Station' if self.config.get('show_legend', False) else None
            )
            
            if self.config.get('label_mt_stations', False):
                for i, (x, y) in enumerate(zip(self.mt_stations['x'][mask], self.mt_stations['y'][mask])):
                    ax.text(x, y, str(i+1), 
                           fontsize=self.config.get('mt_label_fontsize', 6),
                           color='black',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                    facecolor='white', 
                                    alpha=0.7,
                                    edgecolor='none'))
            
        except Exception as e:
            self._log(f"Error adding MT stations: {e}", 'warning')
    
    def _set_km_ticks(self, ax: plt.Axes, x_range: Tuple[float, float], 
                     y_range: Tuple[float, float], is_depth: bool = False) -> None:

        try:
            x_min, x_max = x_range
            x_ticks = np.linspace(x_min, x_max, self.config['n_ticks'])
            
            if x_min <= 0 <= x_max and not np.any(np.abs(x_ticks) < 1e-6):
                zero_idx = np.argmin(np.abs(x_ticks))
                x_ticks[zero_idx] = 0.0
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{x/1000:.0f}' for x in x_ticks], 
                              fontsize=self.config['tick_fontsize'])
            
            y_min, y_max = y_range
            y_ticks = np.linspace(y_min, y_max, self.config['n_ticks'])

            if y_min <= 0 <= y_max and not np.any(np.abs(y_ticks) < 1e-6):
                zero_idx = np.argmin(np.abs(y_ticks))
                y_ticks[zero_idx] = 0.0
            
            ax.set_yticks(y_ticks)
            
            if is_depth:
                tick_labels = []
                for y in y_ticks:
                    if y > 0:
                        tick_labels.append(f'{y/1000:.0f}')
                    elif y < 0:
                        tick_labels.append(f'-{abs(y)/1000:.0f}')  
                    else:
                        tick_labels.append('0')
                ax.set_yticklabels(tick_labels, fontsize=self.config['tick_fontsize'])
            else:
                tick_labels = []
                for y in y_ticks:
                    if y >= 0:
                        tick_labels.append(f'{y/1000:.0f}')
                    else:
                        tick_labels.append(f'-{abs(y)/1000:.0f}')
                ax.set_yticklabels(tick_labels, fontsize=self.config['tick_fontsize'])
                
        except Exception as e:
            self._log(f"Error setting ticks: {e}", 'warning')
    
    def _set_axis_box(self, ax: plt.Axes, x_range: Tuple[float, float], 
                     y_range: Tuple[float, float]) -> None:

        try:
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])
            
        except Exception as e:
            self._log(f"Error setting axis box: {e}", 'warning')
    
    def _align_plot_to_axes(self, ax: plt.Axes, X: np.ndarray, Y: np.ndarray) -> None:

        try:
            x_min, x_max = X.min(), X.max()
            y_min, y_max = Y.min(), Y.max()
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
        except Exception as e:
            self._log(f"Error aligning plot: {e}", 'warning')
    
    def _add_subplot_label(self, ax: plt.Axes, label_idx: int) -> None:
        """
        Add a label (a, b, c, etc.) to a subplot.
        """
        if not self.config['add_subplot_labels']:
            return
        
        try:
            if label_idx < len(self.config['subplot_labels']):
                label_text = self.config['subplot_labels'][label_idx]
            else:
                label_text = f'({chr(97 + label_idx)})'
            
            x_pos, y_pos = self.config['subplot_label_position']

            bbox_props = None
            if self.config['subplot_label_bg'] is not None:
                bbox_props = dict(boxstyle='round,pad=0.3', 
                                 facecolor=self.config['subplot_label_bg'], 
                                 alpha=0.7,
                                 edgecolor='none')

            ax.text(x_pos, y_pos, label_text,
                   transform=ax.transAxes,
                   fontsize=self.config['subplot_label_fontsize'],
                   fontweight=self.config['subplot_label_weight'],
                   color=self.config['subplot_label_color'],
                   bbox=bbox_props,
                   ha='left', va='top',
                   zorder=20)  # High zorder to ensure it's on top
            
        except Exception as e:
            self._log(f"Error adding subplot label: {e}", 'warning')
    
    def plot_horizontal(self, depth: float = 5000, ax: Optional[plt.Axes] = None, 
                       title: Optional[str] = None, plot_stations: Optional[bool] = None, 
                       label_idx: Optional[int] = None, **kwargs) -> Tuple[Optional[plt.Axes], Optional[Any]]:

        if not self.loaded:
            self._log("No data loaded", 'error')
            return None, None
        
        try:
            unique_depths = np.unique(self.z)
            depth_idx = np.argmin(np.abs(unique_depths - depth))
            actual_depth = unique_depths[depth_idx]
            depth_diff = abs(actual_depth - depth)
            
            if depth_diff > self.config['depth_tolerance']:
                self._log(f"Using closest depth: {actual_depth/1000:.1f} km", 'warning')
            
            depth_mask = np.abs(self.z - actual_depth) < self.config['depth_tolerance']
            
            if np.sum(depth_mask) < 4:
                self._log(f"Insufficient data", 'warning')
                return (ax, None) if ax is not None else (None, None)
            
            x_vals = self.x[depth_mask]
            y_vals = self.y[depth_mask]
            rho_vals = self.rho[depth_mask]
            
            X, Y, Z = self._prepare_data_for_plot(
                x_vals, y_vals, rho_vals,
                self.config['x_range'], self.config['y_range']
            )
            
            if X is None:
                self._log(f"No valid data", 'warning')
                return (ax, None) if ax is not None else (None, None)
            
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 6))
            
            plot_config = {**self.config, **kwargs}

            im = ax.pcolormesh(X, Y, Z, 
                              cmap=plot_config['cmap'],
                              vmin=plot_config['vmin'],
                              vmax=plot_config['vmax'],
                              shading='auto',
                              edgecolors=plot_config['pcolormesh_edgecolor'],
                              linewidth=plot_config['pcolormesh_linewidth'],
                              rasterized=True,
                              antialiased=True) 
            
            self._align_plot_to_axes(ax, X, Y)
            
            ax.set_aspect('equal')
            
            if title is None:
                title = f'{actual_depth/1000:.1f} km'
            ax.set_title(title, fontsize=plot_config['title_fontsize'], pad=4)

            self._set_axis_box(ax, self.config['x_range'], self.config['y_range'])
            self._set_km_ticks(ax, self.config['x_range'], self.config['y_range'])
            
            ax.set_xlabel('Easting (km)', fontsize=plot_config['label_fontsize'], labelpad=2)
            ax.set_ylabel('Northing (km)', fontsize=plot_config['label_fontsize'], labelpad=2)
            
            if self.config['grid_visible']:
                ax.grid(True, alpha=self.config['grid_alpha'], 
                       linestyle=self.config['grid_linestyle'], 
                       linewidth=self.config['linewidth'])
            else:
                ax.grid(False)

            for spine in ax.spines.values():
                spine.set_linewidth(self.config['axis_linewidth'])

            ax.tick_params(axis='both', which='both',
                          length=self.config['tick_length'], 
                          width=self.config['tick_width'],
                          top=False, 
                          right=False, 
                          bottom=True, 
                          left=True)  
            
            ax.tick_params(axis='both', which='both', labelsize=self.config['tick_fontsize'])
            
            should_plot_stations = plot_stations if plot_stations is not None else self.config['plot_mt_stations']
            if should_plot_stations and self.mt_stations is not None:
                self._add_mt_stations(ax)
            
            # Add subplot label if index is provided
            if label_idx is not None:
                self._add_subplot_label(ax, label_idx)
            
            return ax, im
            
        except Exception as e:
            self._log(f"Error plotting horizontal slice: {e}", 'error')
            return (ax, None) if ax is not None else (None, None)
    
    def plot_vertical(self, orientation: str = 'x', position: float = 0, 
                     ax: Optional[plt.Axes] = None, title: Optional[str] = None, 
                     plot_stations: bool = False, label_idx: Optional[int] = None, 
                     **kwargs) -> Tuple[Optional[plt.Axes], Optional[Any]]:

        if not self.loaded:
            self._log("No data loaded", 'error')
            return None, None
        
        try:
            orientation = orientation.lower()
            if orientation not in ['x', 'y']:
                self._log(f"Invalid orientation: {orientation}", 'error')
                return None, None
            
            if orientation == 'x':
                mask = np.abs(self.x - position) < self.config['position_tolerance']
                x_vals = self.y[mask]
                y_vals = self.z[mask]
                x_range = self.config['y_range']
                x_label = 'Northing'
                slice_label = f'X = {position/1000:.1f} km'
            else:
                mask = np.abs(self.y - position) < self.config['position_tolerance']
                x_vals = self.x[mask]
                y_vals = self.z[mask]
                x_range = self.config['x_range']
                x_label = 'Easting'
                slice_label = f'Y = {position/1000:.1f} km'
            
            y_range = self.config['z_range']
            
            if np.sum(mask) < 4:
                self._log(f"Insufficient data", 'warning')
                return (ax, None) if ax is not None else (None, None)
            
            rho_vals = self.rho[mask]
            
            X, Y, Z = self._prepare_data_for_plot(x_vals, y_vals, rho_vals, x_range, y_range)
            
            if X is None:
                self._log(f"No valid data", 'warning')
                return (ax, None) if ax is not None else (None, None)
            
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 6))
            
            plot_config = {**self.config, **kwargs}
            
            im = ax.pcolormesh(X, Y, Z,
                              cmap=plot_config['cmap'],
                              vmin=plot_config['vmin'],
                              vmax=plot_config['vmax'],
                              shading='auto',
                              edgecolors=plot_config['pcolormesh_edgecolor'],
                              linewidth=plot_config['pcolormesh_linewidth'],
                              rasterized=True,
                              antialiased=True) 

            self._align_plot_to_axes(ax, X, Y)
            
            ax.set_aspect('equal')
            
            if title is None:
                title = slice_label
            ax.set_title(title, fontsize=plot_config['title_fontsize'], pad=4)
            
            self._set_axis_box(ax, x_range, y_range)
            ax.invert_yaxis()  # Depth increases downward

            self._set_km_ticks(ax, x_range, y_range, is_depth=True)
            
            ax.set_xlabel(f'{x_label} (km)', fontsize=plot_config['label_fontsize'], labelpad=2)
            ax.set_ylabel('Depth (km)', fontsize=plot_config['label_fontsize'], labelpad=2)

            if self.config['grid_visible']:
                ax.grid(True, alpha=self.config['grid_alpha'], 
                       linestyle=self.config['grid_linestyle'], 
                       linewidth=self.config['linewidth'])
            else:
                ax.grid(False)

            for spine in ax.spines.values():
                spine.set_linewidth(self.config['axis_linewidth'])
            
            ax.tick_params(axis='both', which='both',
                          length=self.config['tick_length'], 
                          width=self.config['tick_width'],
                          top=False, 
                          right=False,  
                          bottom=True,  
                          left=True)   
            
            ax.tick_params(axis='both', which='both', labelsize=self.config['tick_fontsize'])

            if label_idx is not None:
                self._add_subplot_label(ax, label_idx)
            
            return ax, im
            
        except Exception as e:
            self._log(f"Error plotting vertical slice: {e}", 'error')
            return (ax, None) if ax is not None else (None, None)
    
    def plot_multiple(self, plots: List[Dict], ncols: int = 3, 
                     figsize: Optional[Tuple[float, float]] = None, 
                     plot_stations: Optional[bool] = None, **kwargs) -> Optional[plt.Figure]:

        if not self.loaded:
            self._log("No data loaded", 'error')
            return None
        
        if not plots:
            self._log("No plots specified", 'error')
            return None
        
        try:
            nplots = len(plots)
            nrows = int(np.ceil(nplots / ncols))
            
            self._log(f"Creating {nrows}×{ncols} grid")
            
            if figsize is None and self.config['figsize'] is None:
                subplot_width = 3.2  
                subplot_height = 2.8  
                
                fig_width = subplot_width * ncols
                fig_height = subplot_height * nrows

                fig_height += 0.8  
                
                figsize = (fig_width, fig_height)
            elif figsize is None:
                figsize = self.config['figsize']
            
            self._log(f"Compact figure size: {figsize[0]:.1f}×{figsize[1]:.1f}")

            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
                'mathtext.fontset': 'stix',
                'axes.linewidth': self.config['spine_width'],
                'axes.labelsize': self.config['label_fontsize'],
                'axes.titlesize': self.config['title_fontsize'],
                'xtick.labelsize': self.config['tick_fontsize'],
                'ytick.labelsize': self.config['tick_fontsize'],
                'xtick.major.width': self.config['tick_width'],
                'ytick.major.width': self.config['tick_width'],
                'xtick.major.size': self.config['tick_length'],
                'ytick.major.size': self.config['tick_length'],
            })
            
            fig = plt.figure(figsize=figsize, constrained_layout=False)

            left = 0.06  
            right = 0.98 
            top = 0.94 
            bottom = self.config['cbar_y_position'] + self.config['cbar_height'] + 0.08  # Reduced

            gs = fig.add_gridspec(
                nrows=nrows, ncols=ncols,
                hspace=self.config['hspace'], 
                wspace=self.config['wspace'], 
                top=top,
                bottom=bottom,
                left=left,
                right=right
            )
            
            self._log(f"Minimal spacing: wspace={self.config['wspace']:.3f}, hspace={self.config['hspace']:.3f}")
            
            images = []
            axes_list = []
            
            last_row_for_column = {}
            for col in range(ncols):
                max_row = -1
                for idx, plot_config in enumerate(plots):
                    plot_row = idx // ncols
                    plot_col = idx % ncols
                    if plot_col == col:
                        max_row = max(max_row, plot_row)
                last_row_for_column[col] = max_row
            
            for idx, plot_config in enumerate(plots):
                row = idx // ncols
                col = idx % ncols
                
                ax = fig.add_subplot(gs[row, col])
                plot_type = plot_config.get('type', 'horizontal')
                
                subplot_plot_stations = plot_config.get('plot_stations')
                if subplot_plot_stations is None:
                    subplot_plot_stations = plot_stations
                
                if subplot_plot_stations is None:
                    subplot_plot_stations = self.config['plot_mt_stations']
                
                if plot_type == 'horizontal':
                    ax, im = self.plot_horizontal(
                        depth=plot_config.get('depth', 5000),
                        ax=ax,
                        title=plot_config.get('title', None),
                        plot_stations=subplot_plot_stations,
                        label_idx=idx, 
                        **kwargs
                    )
                elif plot_type == 'vertical':
                    ax, im = self.plot_vertical(
                        orientation=plot_config.get('orientation', 'x'),
                        position=plot_config.get('position', 0),
                        ax=ax,
                        title=plot_config.get('title', None),
                        label_idx=idx,  
                        **kwargs  
                    )
                else:
                    self._log(f"Invalid plot type '{plot_type}'", 'warning')
                    im = None
                
                if im is not None:
                    images.append(im)
                    axes_list.append(ax)

                    if col == 0:
                        ax.set_ylabel(ax.get_ylabel(), fontsize=self.config['label_fontsize'], labelpad=1)
                        ax.tick_params(axis='y', labelleft=True, labelsize=self.config['tick_fontsize'])
                    else:
                        ax.set_ylabel('')
                        ax.tick_params(axis='y', labelleft=False) 

                    is_bottom_row_of_grid = (row == nrows - 1)
                    is_last_row_of_column = (row == last_row_for_column[col])
                    
                    if is_bottom_row_of_grid or is_last_row_of_column:
                        ax.set_xlabel(ax.get_xlabel(), fontsize=self.config['label_fontsize'], labelpad=1)
                        ax.tick_params(axis='x', labelbottom=True, labelsize=self.config['tick_fontsize'])
                    else:
                        ax.set_xlabel('')
                        ax.tick_params(axis='x', labelbottom=False)  
                    
                    ax.tick_params(axis='both', which='both',
                                  length=self.config['tick_length'],
                                  width=self.config['tick_width'],
                                  top=False,    
                                  right=False, 
                                  bottom=True, 
                                  left=True)    
                    
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10, color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    axes_list.append(ax)
            
            for idx in range(nplots, nrows * ncols):
                row = idx // ncols
                col = idx % ncols
                ax = fig.add_subplot(gs[row, col])
                ax.set_visible(False)
            
            if images:
                self._add_compact_colorbar(fig, images[0])
            else:
                self._log("No valid images", 'warning')
            
            plt.subplots_adjust(
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                wspace=self.config['wspace'],
                hspace=self.config['hspace']
            )
            
            self._log("Compact grid plot created", 'success')
            return fig
            
        except Exception as e:
            self._log(f"Error creating multiple plots: {e}", 'error')
            return None
    
    def _add_compact_colorbar(self, fig: plt.Figure, im) -> None:

        try:
            cbar_left = 0.5 - (self.config['cbar_width'] / 2)
            cbar_bottom = self.config['cbar_y_position']
            cbar_width = self.config['cbar_width']
            cbar_height = self.config['cbar_height']
            
            self._log(f"Compact colorbar: {cbar_width:.2f}×{cbar_height:.3f}")
            
            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(self.config['colorbar_label'], 
                          fontsize=self.config['fontsize'] - 1, 
                          labelpad=4) 

            if self.config['colorbar_ticks']:
                cbar.set_ticks(self.config['colorbar_ticks'])
            
            cbar.ax.tick_params(labelsize=self.config['tick_fontsize'] - 1, 
                               length=2, 
                               width=0.6) 

            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            cbar.ax.xaxis.set_major_formatter(formatter)

            cbar.outline.set_linewidth(1.0)
            
        except Exception as e:
            self._log(f"Error adding colorbar: {e}", 'warning')
    
    def save_figure(self, fig: plt.Figure, filename: str, **kwargs) -> bool:

        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not Path(filename).suffix:
                filename = f"{filename}.{self.config['save_format']}"
            
            output_path = output_dir / filename
            
            # Compact save settings
            save_config = {
                'dpi': self.config['save_dpi'],
                'bbox_inches': 'tight',
                'pad_inches': 0.02,  
                'facecolor': 'white',
                'edgecolor': 'none',
                'transparent': False,
            }
            save_config.update(kwargs)
            
            fig.savefig(str(output_path), **save_config)
            self._log(f"Saved: {output_path}", 'success')
            return True
            
        except Exception as e:
            self._log(f"Error saving figure: {e}", 'error')
            return False
    
    def create_demo_data(self, output_file: str = 'demo_data.csv', n_points: int = 10000) -> str:

        try:
            np.random.seed(42)
            
            x = np.random.uniform(-50000, 50000, n_points)
            y = np.random.uniform(-60000, 60000, n_points)
            z = np.random.uniform(0, 40000, n_points)
            
            # Avoid exact zeros to prevent white lines
            x = np.where(np.abs(x) < 100, x + 200, x)
            y = np.where(np.abs(y) < 100, y + 200, y)
            
            r = np.sqrt(x**2 + y**2)
            rho = 2.0 + 3.0 * np.exp(-r**2 / 1e9) * np.exp(-z / 15000)
            rho += np.random.normal(0, 0.2, n_points)
            
            data = np.column_stack([x, y, z, rho])
            header = "x,y,z,rho"
            
            output_path = Path(output_file)
            np.savetxt(output_path, data, delimiter=',', header=header, comments='')
            
            self._log(f"Demo data created: {output_path}", 'success')
            return str(output_path)
            
        except Exception as e:
            self._log(f"Error creating demo data: {e}", 'error')
            return ""

# USAGE EXAMPLE
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    plotter = ModemPlotter('output/gvb_inv5b_NLCG_130.csv', {
        'x_range_km': (-60, 60), 
        'y_range_km': (-70, 70),
        'output_dir': 'output', 
        'dpi': 600,
        'cbar_y_position': 0.08, 
        'wspace': 0.05, 
        'hspace': 0.01, 
        'save_dpi': 600,
        'cmap': 'jet_r',
        'vmin': 0.0,
        'vmax': 5.0,
        
        # Set this to True since your data is in natural log
        'data_in_natural_log': True,
        'convert_to_log10': True,
        
        # MT station configuration
        'mt_station_file': 'mt_stations.txt',  # MT station file
        'plot_mt_stations': False,  
        'mt_station_color': 'white',
        'mt_station_marker': '^',
        'mt_station_size': 45,
        'mt_station_edgecolor': 'black',
        'mt_station_linewidth': 0.5,
        'mt_station_alpha': 0.9,
        'mt_units': 'km',
        
        # Visualization settings
        'grid_visible': False,
        'zero_line_visible': False,
        
        # Subplot label settings
        'add_subplot_labels': True,
        'subplot_labels': ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],  # Custom labels
        'subplot_label_fontsize': 14,
        'subplot_label_weight': 'bold',
        'subplot_label_position': (0.02, 0.98), 
        'subplot_label_color': 'black',
        'subplot_label_bg': None, 
    })
    
    # Define depths
    depths = [1, 5, 10, 20, 30, 40]
    
    plots = []
    for i, depth in enumerate(depths):
        plot_config = {
            'type': 'horizontal', 
            'depth': depth*1000, 
            'title': f'Depth ={depth} km b.s.l'
        }
        
        if i == 1:  
            plot_config['plot_stations'] = True  
        else:
            plot_config['plot_stations'] = False  
        
        plots.append(plot_config)

    fig = plotter.plot_multiple(plots, ncols=3, figsize=(10, 10))
    
    if fig:
        plotter.save_figure(fig, 'depth_slices_mt_second_only.png')
        plt.show()