"""
Lunar Landing Trajectory Optimization using Direct Collocation Method

This module implements a direct collocation approach for solving the lunar landing
optimal control problem. The spacecraft starts at 15 km altitude with horizontal
velocity of 1691.92 m/s and must land with zero horizontal velocity and -5 m/s
vertical velocity while minimizing fuel consumption (flight time).

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class LunarLandingOptimizer:
    """
    A class to solve the lunar landing trajectory optimization problem using
    direct collocation method.
    """
    
    def __init__(self):
        """Initialize the lunar landing optimizer with problem parameters."""
        # Problem parameters
        self.params = {
            'T': 440.0,          # Thrust (N)
            'M0': 300.0,         # Initial mass (kg)
            'mdot': 0.15,        # Mass flow rate (kg/s)
            'mu': 4.9026e12,     # Lunar gravitational parameter (m^3/s^2)
            'RM': 1.7382e6       # Lunar radius (m)
        }
        
        # Discretization parameters
        self.nNode = 31          # Number of collocation nodes
        self.nX = 4              # Number of state variables
        self.nXU = 5             # Number of state + control variables
        self.nNLP = self.nXU * self.nNode + 1  # Total optimization variables
        
        # Scaling parameters for numerical conditioning
        self._setup_scaling()
        
        # Create results directory if it doesn't exist
        os.makedirs('./results', exist_ok=True)
    
    def _setup_scaling(self):
        """Setup scaling parameters for better numerical conditioning."""
        # Variable bounds for scaling
        minval = np.array([self.params['RM'], 0, -10, -100, np.pi/2])
        maxval = np.array([self.params['RM']+25000, np.pi, 2000, 50, np.pi])
        
        # Scaling coefficients
        self.Scale_k = np.zeros(self.nXU)
        self.Scale_b = np.zeros(self.nXU)
        
        for i in range(self.nXU):
            self.Scale_k[i] = 1.0 / (maxval[i] - minval[i])
            self.Scale_b[i] = 0.5 - maxval[i] / (maxval[i] - minval[i])
        
        self.Scale_tf = 0.5 / 1000.0
    
    def _create_initial_guess(self) -> np.ndarray:
        """
        Create initial guess for optimization variables.
        
        Returns:
            np.ndarray: Initial guess vector
        """
        # Initial and final state + control guess
        v1 = np.array([self.params['RM']+15000, np.pi/2, 1691.92, 0, 2*np.pi/3])
        v2 = np.array([self.params['RM'], 0, 0, -5, np.pi/2])
        
        # Scale the values
        v1_scaled = v1 * self.Scale_k + self.Scale_b
        v2_scaled = v2 * self.Scale_k + self.Scale_b
        dv = v2_scaled - v1_scaled
        
        # Create initial guess vector
        x0 = np.zeros(self.nNLP)
        
        for j in range(self.nNode):
            for i in range(self.nXU):
                x0[j + i * self.nNode] = v1_scaled[i] + (j) * dv[i] / (self.nNode - 1)
        
        x0[self.nNLP-1] = 900 * self.Scale_tf  # Initial guess for final time
        
        return x0
    
    def objective(self, x: np.ndarray) -> float:
        """
        Objective function: minimize final time (fuel consumption).
        
        Args:
            x: Optimization variables
            
        Returns:
            float: Objective value (final time)
        """
        return x[-1]  # Minimize final time
    
    def _extract_variables(self, x: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Extract and unscale optimization variables.
        
        Args:
            x: Scaled optimization variables
            
        Returns:
            Tuple containing r, phi, u, v, beta, tf
        """
        tf = x[self.nNLP-1] / self.Scale_tf
        
        r = (x[0:self.nNode] - self.Scale_b[0]) / self.Scale_k[0]
        phi = (x[self.nNode:2*self.nNode] - self.Scale_b[1]) / self.Scale_k[1]
        u = (x[2*self.nNode:3*self.nNode] - self.Scale_b[2]) / self.Scale_k[2]
        v = (x[3*self.nNode:4*self.nNode] - self.Scale_b[3]) / self.Scale_k[3]
        beta = (x[4*self.nNode:5*self.nNode] - self.Scale_b[4]) / self.Scale_k[4]
        
        return r, phi, u, v, beta, tf
    
    def _dynamics(self, r: float, phi: float, u: float, v: float, 
                  beta: float, m: float) -> Tuple[float, ...]:
        """
        Compute state derivatives (dynamics).
        
        Args:
            r, phi, u, v: State variables
            beta: Control variable
            m: Mass
            
        Returns:
            Tuple of state derivatives
        """
        dr = v
        dphi = u / r
        du = -u*v/r + self.params['T']*np.cos(beta)/m
        dv = u**2/r - self.params['mu']/r**2 + self.params['T']*np.sin(beta)/m
        
        return dr, dphi, du, dv
    
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Compute all constraint violations.
        
        Args:
            x: Optimization variables
            
        Returns:
            np.ndarray: Constraint violations
        """
        r, phi, u, v, beta, tf = self._extract_variables(x)
        h = tf / (self.nNode - 1)  # Time step
        
        constraints = []
        
        # Initial conditions
        constraints.extend([
            r[0] - (self.params['RM'] + 15000),
            phi[0] - np.pi/2,
            u[0] - 1691.92,
            v[0] - 0
        ])
        
        # Final conditions
        constraints.extend([
            r[self.nNode-1] - self.params['RM'],
            u[self.nNode-1] - 0,
            v[self.nNode-1] - (-5)
        ])
        
        # Collocation constraints (trapezoidal rule)
        for k in range(self.nNode-1):
            t_k = k * h
            t_kp1 = (k + 1) * h
            
            m_k = self.params['M0'] - self.params['mdot'] * t_k
            m_kp1 = self.params['M0'] - self.params['mdot'] * t_kp1
            
            # Skip if mass becomes negative
            if m_k <= 0 or m_kp1 <= 0:
                continue
            
            # Dynamics at k and k+1
            dr_k, dphi_k, du_k, dv_k = self._dynamics(r[k], phi[k], u[k], v[k], beta[k], m_k)
            dr_kp1, dphi_kp1, du_kp1, dv_kp1 = self._dynamics(r[k+1], phi[k+1], u[k+1], v[k+1], beta[k+1], m_kp1)
            
            # Trapezoidal integration constraints
            constraints.extend([
                r[k+1] - r[k] - h/2 * (dr_k + dr_kp1),
                phi[k+1] - phi[k] - h/2 * (dphi_k + dphi_kp1),
                u[k+1] - u[k] - h/2 * (du_k + du_kp1),
                v[k+1] - v[k] - h/2 * (dv_k + dv_kp1)
            ])
        
        return np.array(constraints)
    
    def inequality_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Compute inequality constraint violations.
        
        Args:
            x: Optimization variables
            
        Returns:
            np.ndarray: Inequality constraint violations
        """
        r, phi, u, v, beta, tf = self._extract_variables(x)
        h = tf / (self.nNode - 1)
        
        ineq_constraints = []
        
        # Path constraints
        for k in range(self.nNode):
            # Altitude must be non-negative
            ineq_constraints.append(self.params['RM'] - r[k])  # r >= RM
            
            # Mass must be positive
            t_k = k * h
            m_k = self.params['M0'] - self.params['mdot'] * t_k
            ineq_constraints.append(10 - m_k)  # m >= 10 kg
        
        return np.array(ineq_constraints)
    
    def optimize(self) -> Dict[str, Any]:
        """
        Solve the lunar landing optimization problem.
        
        Returns:
            Dict containing optimization results
        """
        print("Starting Lunar Landing Trajectory Optimization...")
        print("Using Direct Collocation Method with {} nodes".format(self.nNode))
        print("="*60)
        
        # Create initial guess
        x0 = self._create_initial_guess()
        
        # Bounds: all variables scaled to [-0.5, 0.5]
        bounds = [(-0.5, 0.5) for _ in range(self.nNLP)]
        
        # Constraint dictionaries
        eq_constraint = {'type': 'eq', 'fun': self.constraints}
        ineq_constraint = {'type': 'ineq', 'fun': lambda x: -self.inequality_constraints(x)}
        
        # Optimization options
        options = {
            'maxiter': 3000,
            'ftol': 1e-6,
            'disp': True
        }
        
        # Try SLSQP method first
        print("Trying SLSQP optimization method...")
        result = minimize(
            self.objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[eq_constraint, ineq_constraint],
            options=options
        )
        
        # If SLSQP fails, try trust-constr
        if not result.success:
            print("\nSLSQP failed, trying trust-constr method...")
            result = minimize(
                self.objective, x0,
                method='trust-constr',
                bounds=bounds,
                constraints=[eq_constraint, ineq_constraint],
                options={'maxiter': 1000, 'disp': True}
            )
        
        return result
    
    def print_results(self, result: Dict[str, Any]):
        """
        Print detailed optimization results.
        
        Args:
            result: Optimization result from scipy.optimize
        """
        if not result.success:
            print("Optimization failed!")
            return
        
        # Extract solution
        r, phi, u, v, beta, tf_opt = self._extract_variables(result.x)
        
        # Calculate final state
        final_altitude = (r[-1] - self.params['RM']) / 1000  # km
        final_mass = self.params['M0'] - self.params['mdot'] * tf_opt
        
        # Calculate boundary condition errors
        pos_err = abs(final_altitude)
        u_err = abs(u[-1] - 0)
        v_err = abs(v[-1] - (-5))
        
        # Print results
        print("\n" + "="*40)
        print("      OPTIMIZATION RESULTS")
        print("="*40)
        print(f"Success: {result.success}")
        print(f"Exit message: {result.message}")
        print(f"Final objective value: {result.fun:.6e}")
        print(f"Optimal flight time: {tf_opt:.2f} seconds")
        print(f"Final mass: {final_mass:.2f} kg")
        print(f"Number of iterations: {result.nit}")
        
        print("\n" + "="*40)
        print("         FINAL STATE")
        print("="*40)
        print(f"Final altitude: {final_altitude:.6f} km (Target: 0 km)")
        print(f"Final horizontal velocity: {u[-1]:.6f} m/s (Target: 0.0 m/s)")
        print(f"Final vertical velocity: {v[-1]:.6f} m/s (Target: -5.0 m/s)")
        print(f"Final mass: {final_mass:.2f} kg")
        
        print(f"\nBoundary Condition Errors:")
        print(f"  Position:    {pos_err:.2e} km")
        print(f"  Horizontal:  {u_err:.2e} m/s")
        print(f"  Vertical:    {v_err:.2e} m/s")
        print("="*40)
    
    def plot_results(self, result: Dict[str, Any]):
        """
        Create and save beautiful plots of the optimization results.
        
        Args:
            result: Optimization result from scipy.optimize
        """
        if not result.success:
            print("Cannot plot results - optimization failed!")
            return
        
        # Extract solution
        r, phi, u, v, beta, tf_opt = self._extract_variables(result.x)
        
        # Time vector
        t = np.linspace(0, tf_opt, self.nNode)
        
        # Convert to plotting units
        altitude = (r - self.params['RM']) / 1000  # km
        phi_deg = phi * 180/np.pi
        beta_deg = beta * 180/np.pi
        mass = self.params['M0'] - self.params['mdot'] * t
        
        # Define beautiful colors
        colors = ['#3366CC', '#CC3333', '#33CC66', '#FF9933', '#CC33CC', '#666666']
        
        # Create main figure with state and control plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor('white')
        
        # Altitude plot
        axes[0,0].plot(t, altitude, color=colors[0], linewidth=2.5)
        axes[0,0].set_xlabel('Time (s)', fontweight='bold')
        axes[0,0].set_ylabel('Altitude (km)', fontweight='bold')
        axes[0,0].set_title('Altitude Profile', fontweight='bold', fontsize=14)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_xlim([0, tf_opt])
        axes[0,0].set_ylim([0, max(altitude)*1.05])
        
        # Angle plot
        axes[0,1].plot(t, phi_deg, color=colors[1], linewidth=2.5)
        axes[0,1].set_xlabel('Time (s)', fontweight='bold')
        axes[0,1].set_ylabel('Angle φ (deg)', fontweight='bold')
        axes[0,1].set_title('Flight Path Angle', fontweight='bold', fontsize=14)
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_xlim([0, tf_opt])
        
        # Horizontal velocity plot
        axes[0,2].plot(t, u, color=colors[2], linewidth=2.5)
        axes[0,2].set_xlabel('Time (s)', fontweight='bold')
        axes[0,2].set_ylabel('Horizontal Velocity u (m/s)', fontweight='bold')
        axes[0,2].set_title('Horizontal Velocity', fontweight='bold', fontsize=14)
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].set_xlim([0, tf_opt])
        
        # Vertical velocity plot
        axes[1,0].plot(t, v, color=colors[3], linewidth=2.5)
        axes[1,0].axhline(y=-5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[1,0].text(tf_opt*0.7, -5.5, 'Target: -5 m/s', fontsize=10, color='gray')
        axes[1,0].set_xlabel('Time (s)', fontweight='bold')
        axes[1,0].set_ylabel('Vertical Velocity v (m/s)', fontweight='bold')
        axes[1,0].set_title('Vertical Velocity', fontweight='bold', fontsize=14)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_xlim([0, tf_opt])
        
        # Thrust angle plot
        axes[1,1].plot(t, beta_deg, color=colors[4], linewidth=2.5)
        axes[1,1].set_xlabel('Time (s)', fontweight='bold')
        axes[1,1].set_ylabel('Thrust Angle β (deg)', fontweight='bold')
        axes[1,1].set_title('Control Input', fontweight='bold', fontsize=14)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlim([0, tf_opt])
        
        # Mass plot
        axes[1,2].plot(t, mass, color=colors[5], linewidth=2.5)
        axes[1,2].set_xlabel('Time (s)', fontweight='bold')
        axes[1,2].set_ylabel('Mass (kg)', fontweight='bold')
        axes[1,2].set_title('Spacecraft Mass', fontweight='bold', fontsize=14)
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].set_xlim([0, tf_opt])
        
        # Main title
        fig.suptitle(f'Lunar Landing Trajectory - Direct Collocation Method\n'
                    f'(tf = {tf_opt:.2f} s, Final Mass = {mass[-1]:.2f} kg)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('./results/lunar_landing_states.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create trajectory plot
        fig2, ax = plt.subplots(figsize=(10, 8))
        fig2.patch.set_facecolor('white')
        
        # Calculate trajectory coordinates
        x_traj = r * np.cos(phi) / 1000  # km
        y_traj = r * np.sin(phi) / 1000  # km
        
        # Plot trajectory
        ax.plot(x_traj, y_traj, color='#3366CC', linewidth=3, label='Trajectory')
        ax.plot(x_traj[0], y_traj[0], 'go', markersize=12, markerfacecolor='green', 
                markeredgewidth=2, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=12, markerfacecolor='red', 
                markeredgewidth=2, label='Landing')
        
        # Add Moon surface arc
        theta_moon = np.linspace(0, np.pi/2, 100)
        x_moon = self.params['RM'] * np.cos(theta_moon) / 1000
        y_moon = self.params['RM'] * np.sin(theta_moon) / 1000
        ax.plot(x_moon, y_moon, 'k-', linewidth=2)
        ax.fill_between(x_moon, 0, y_moon, color='gray', alpha=0.3, label='Moon Surface')
        
        # Formatting
        ax.set_xlabel('Horizontal Distance (km)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Vertical Distance (km)', fontweight='bold', fontsize=14)
        ax.set_title('Lunar Landing Trajectory in 2D Space', fontweight='bold', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add annotations
        ax.annotate(f'Start\n(h={altitude[0]:.1f} km)', 
                   xy=(x_traj[0], y_traj[0]), xytext=(x_traj[0]+50, y_traj[0]+50),
                   fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', edgecolor='black'))
        ax.annotate(f'Landing\n(v={v[-1]:.1f} m/s)', 
                   xy=(x_traj[-1], y_traj[-1]), xytext=(x_traj[-1]+50, y_traj[-1]+100),
                   fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig('./results/lunar_landing_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nPlots saved to ./results/ directory")


def main():
    """Main function to run the lunar landing optimization."""
    # Create optimizer instance
    optimizer = LunarLandingOptimizer()
    
    # Solve the optimization problem
    result = optimizer.optimize()
    
    # Print results
    optimizer.print_results(result)
    
    # Create plots
    optimizer.plot_results(result)


if __name__ == "__main__":
    main()