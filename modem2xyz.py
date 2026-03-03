"""
Author: Hakim Rezayee
Date: January 22, 2026
Affiliation: Geoscience Enterprise Inc.
Description: Script to extract the resistivity model result from ModEM (.rho) to xyz format (.txt and .csv).

Note: To display the resistivity model slices in depths or vertical sections use the modem_plotter. 

Please do not change anything if not necessary!

"""

import numpy as np
import os

def to_csv(rho_file, output_file=None):
    """Convert ModEM to CSV/TXT - KEEP ORIGINAL RESISTIVITY VALUES"""
    # Read file
    with open(rho_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    
    # Get dimensions
    ny, nx, nz = map(int, lines[0].split()[:3])
    is_log = 'LOGE' in lines[0].upper()
    
    # Get all numbers
    all_nums = []
    for line in lines[1:]:
        all_nums.extend(map(float, line.split()))
    
    # Parse
    idx = 0
    dy = all_nums[idx:idx+ny]; idx += ny
    dx = all_nums[idx:idx+nx]; idx += nx
    dz = all_nums[idx:idx+nz]; idx += nz
    rho = np.array(all_nums[idx:idx+ny*nx*nz]).reshape((ny, nx, nz), order='F')
    
    x_edges = [-sum(dx)/2]
    for d in dx:
        x_edges.append(x_edges[-1] + d)
    x_centers = [(x_edges[i] + x_edges[i+1])/2 for i in range(len(x_edges)-1)]
    
    # Y (flip for north)
    y_edges = [-sum(dy)/2]
    for d in dy:
        y_edges.append(y_edges[-1] + d)
    y_edges = [-y for y in y_edges]
    y_centers = [(y_edges[i] + y_edges[i+1])/2 for i in range(len(y_edges)-1)]
    
    # Z (0 at highest elevation)
    z_edges = [0.0]
    for d in dz:
        z_edges.append(z_edges[-1] + d)
    z_centers = [(z_edges[i] + z_edges[i+1])/2 for i in range(len(z_edges)-1)]
    
    # Create output folder
    os.makedirs("output", exist_ok=True)
    
    # Set output name
    if output_file is None:
        base = os.path.splitext(os.path.basename(rho_file))[0]
        csv_file = f"output/{base}.csv"
        txt_file = f"output/{base}.txt"
    else:
        csv_file = f"output/{output_file}.csv"
        txt_file = f"output/{output_file}.txt"
    
    # Write CSV
    with open(csv_file, 'w') as f:
        f.write("x,y,z,resistivity\n")
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f"{x_centers[ix]:.6e},{y_centers[iy]:.6e},{z_centers[iz]:.6e},{rho[iy,ix,iz]:.6e}\n")
    
    # Write TXT
    with open(txt_file, 'w') as f:
        f.write("x y z resistivity\n")
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f"{x_centers[ix]:.6e} {y_centers[iy]:.6e} {z_centers[iz]:.6e} {rho[iy,ix,iz]:.6e}\n")
    
    print(f"Created: {csv_file}")
    print(f"Created: {txt_file}")
    print(f"Total points: {nx*ny*nz:,}")
    print(f"Resistivity is {'in natural logarithm' if is_log else 'linear'}")
    
    return csv_file, txt_file