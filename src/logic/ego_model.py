import casadi as ca
# Casadi based kinematic Model originally from https://github.com/dotchen/WorldOnRails/blob/release/rails/models/ego_model.py
class EgoModel:
    def __init__(self, dt=1. / 4,L=2.5, Cd=0.3, A=2.2, rho=1.225, Cr=0.015, m=1200):
        self.dt = dt

        # Kinematic bicycle model parameters, tuned from World on Rails
        self.front_wb = -0.090769015
        self.rear_wb = 1.4178275
        self.steer_gain = 0.36848336

    # Parameters for drag force and rolling resistance
        self.Cd = Cd          # Aerodynamic drag coefficient
        self.A = A            # Frontal area (m^2)
        self.rho = rho        # Air density (kg/m^3)
        self.Cr = Cr          # Rolling resistance coefficient
        self.m = m            # Vehicle mass (kg)

    def forward(self, locs, yaws, spds, acts):
        steer = acts[0]
        accel = acts[1] + 0.2  # Acceleration can be positive or negative

        # Calculate steering dynamics
        wheel = self.steer_gain * steer
        beta = ca.atan(self.rear_wb / (self.front_wb + self.rear_wb) * ca.tan(wheel))

        # Calculate resistive forces
        F_drag = 0.5 * self.rho * self.Cd * self.A * spds**2  # Aerodynamic drag
        F_rolling = self.m * 9.81 * self.Cr  # Rolling resistance (flat road assumed)

        # Effective acceleration after resistive forces
        accel_effective = accel - (F_drag + F_rolling) / self.m

        # Update the state
        next_locs = locs + spds * ca.vertcat(ca.cos(yaws + beta), ca.sin(yaws + beta)) * self.dt
        next_yaws = yaws + spds / self.rear_wb * ca.sin(beta) * self.dt
        next_spds = spds + accel_effective * self.dt

        # Ensure speed does not go below zero using CasADi's fmax function
        next_spds = ca.fmax(next_spds, 0)

        return next_locs, next_yaws, next_spds
