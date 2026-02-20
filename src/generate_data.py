"""
Generate synthetic dataset using OpenSeesPy for a 2-story steel frame.
Varies structural parameters and ground motion scale factor.
Outputs: roof displacement, inter-story drift, floor acceleration.
"""

import numpy as np
import pandas as pd
import openseespy.opensees as ops
import os
import time
from scipy.stats import qmc

def generate_sine_burst(dt=0.01, duration=20.0, freq=2.0, scale=1.0):
    """Create a simple sine burst acceleration time series (m/s²)."""
    t = np.arange(0, duration, dt)
    envelope = np.exp(-0.5 * ((t - 5.0) / 2.0) ** 2)
    accel = scale * envelope * np.sin(2 * np.pi * freq * t)
    return t, accel

def run_simulation(params):
    """
    params: dict with keys: E, I_col, mass, damping_ratio, scale
    Returns dict of max responses.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # Nodes
    ops.node(1, 0.0, 0.0)
    ops.node(2, 6.0, 0.0)
    ops.node(3, 0.0, 3.0)
    ops.node(4, 6.0, 3.0)
    ops.node(5, 0.0, 6.0)
    ops.node(6, 6.0, 6.0)

    # Fix base
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)

    # Material
    A = 0.01  # m²
    ops.uniaxialMaterial('Elastic', 1, params['E'])

    # Transformation
    ops.geomTransf('Linear', 1)

    # Columns
    ops.element('elasticBeamColumn', 1, 1, 3, A, params['E'], params['I_col'], 1)
    ops.element('elasticBeamColumn', 2, 2, 4, A, params['E'], params['I_col'], 1)
    ops.element('elasticBeamColumn', 3, 3, 5, A, params['E'], params['I_col'], 1)
    ops.element('elasticBeamColumn', 4, 4, 6, A, params['E'], params['I_col'], 1)

    # Beams
    ops.element('elasticBeamColumn', 5, 3, 4, A, params['E'], params['I_col'], 1)
    ops.element('elasticBeamColumn', 6, 5, 6, A, params['E'], params['I_col'], 1)

    # Masses (lumped at floor nodes)
    ops.mass(3, params['mass'], params['mass'], 0.0)
    ops.mass(4, params['mass'], params['mass'], 0.0)
    ops.mass(5, params['mass']/2, params['mass']/2, 0.0)
    ops.mass(6, params['mass']/2, params['mass']/2, 0.0)

    # Eigen analysis for first mode frequency (for damping)
    eigen_values = ops.eigen(2)   # use default, faster solver
    omega1 = np.sqrt(eigen_values[0])
    # Rayleigh damping (mass proportional only)
    a0 = 2 * params['damping_ratio'] * omega1
    ops.rayleigh(a0, 0.0, 0.0, 0.0)

    # Ground motion
    dt = 0.01
    t, accel = generate_sine_burst(dt=dt, scale=params['scale'])
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *accel)
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1)

    # Recorders
    ops.recorder('Node', '-file', 'disp_node3.out', '-time', '-node', 3, '-dof', 1, 'disp')
    ops.recorder('Node', '-file', 'disp_node5.out', '-time', '-node', 5, '-dof', 1, 'disp')
    ops.recorder('Node', '-file', 'accel_node5.out', '-time', '-node', 5, '-dof', 1, 'accel')

    # Analysis
    ops.wipeAnalysis()
    ops.system('BandSPD')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.algorithm('Linear')
    ops.analysis('Transient')

    ok = ops.analyze(len(accel), dt)
    if ok != 0:
        print(f"Analysis failed for params: {params}")
        # Clean up and return NaNs
        for f in ['disp_node3.out', 'disp_node5.out', 'accel_node5.out']:
            if os.path.exists(f):
                os.remove(f)
        ops.wipe()
        return {'roof_disp': np.nan, 'drift_max': np.nan, 'accel_max': np.nan}

    # Read results
    disp3 = np.loadtxt('disp_node3.out')
    disp5 = np.loadtxt('disp_node5.out')
    accel5 = np.loadtxt('accel_node5.out')

    # Compute responses
    roof_disp = np.max(np.abs(disp5[:, 1]))          # max roof displacement
    drift1 = np.abs(disp3[:, 1]) / 3.0               # first story drift ratio
    drift2 = np.abs(disp5[:, 1] - disp3[:, 1]) / 3.0 # second story drift ratio
    drift_max = max(np.max(drift1), np.max(drift2))
    accel_max = np.max(np.abs(accel5[:, 1]))

    # Clean up with retry
    for f in ['disp_node3.out', 'disp_node5.out', 'accel_node5.out']:
       for attempt in range(3):  # try up to 3 times
            try:
                if os.path.exists(f):
                   os.remove(f)
                break
            except PermissionError:
                  time.sleep(0.1)  # wait a bit and retry
    ops.wipe()

    return {'roof_disp': roof_disp, 'drift_max': drift_max, 'accel_max': accel_max}

def main(n_samples=500):
    param_ranges = {
        'E': (190e9, 210e9),
        'I_col': (1e-4, 5e-4),
        'mass': (5000, 15000),
        'damping_ratio': (0.02, 0.05),
        'scale': (0.5, 2.0)
    }

    sampler = qmc.LatinHypercube(d=len(param_ranges))
    sample = sampler.random(n=n_samples)
    scaled = qmc.scale(sample,
                       [v[0] for v in param_ranges.values()],
                       [v[1] for v in param_ranges.values()])

    data = []
    param_names = list(param_ranges.keys())

    for i in range(n_samples):
        params = dict(zip(param_names, scaled[i, :]))
        print(f"Running sample {i+1}/{n_samples}: {params}")
        responses = run_simulation(params)
        row = {**params, **responses}
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv('data/raw_dataset.csv', index=False)
    print(f"Dataset saved with {len(df)} rows.")

if __name__ == '__main__':
    main(500)