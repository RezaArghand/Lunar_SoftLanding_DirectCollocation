# Lunar Landing Trajectory Optimization

A Python implementation of lunar landing trajectory optimization using the direct collocation method. This project solves the optimal control problem of landing a spacecraft on the Moon's surface while minimizing fuel consumption.

## Problem Description

The spacecraft starts at an altitude of 15 km above the Moon's surface with:
- Initial horizontal velocity: 1691.92 m/s
- Initial vertical velocity: 0 m/s
- Initial mass: 300 kg

The objective is to land the spacecraft with:
- Final horizontal velocity: 0 m/s
- Final vertical velocity: -5 m/s
- Minimize fuel consumption (equivalent to minimizing flight time)

## Method

The optimization problem is solved using the **Direct Collocation Method**:
- Discretizes the continuous optimal control problem into a nonlinear programming (NLP) problem
- Uses trapezoidal rule for numerical integration of the dynamics
- Employs variable scaling for better numerical conditioning
- Solves the resulting NLP using Sequential Least Squares Programming (SLSQP)

## Features

- **Robust Optimization**: Multiple solver attempts (SLSQP, Trust-Constr)
- **Beautiful Visualizations**: Professional-quality plots of trajectory and states
- **Detailed Results**: Comprehensive output of optimization results and boundary condition errors
- **Modular Design**: Object-oriented implementation for easy extension
- **Result Storage**: Automatic saving of plots to results directory

## Requirements
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0

python lunar_landing_optimization.py

lunar-landing-optimization/
│
├── lunar_landing_optimization.py  # Main optimization code
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore file
└── results/                      # Generated plots and results
    ├── lunar_landing_states.png
    └── lunar_landing_trajectory.png

Reza Arghand
Arghand.eng@gmail.com

    

