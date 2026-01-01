# Copyright (c) 2025, MAHDYAR KARIMI
# Author: MAHDYAR KARIMI, 2025-08-31 

def IDM(Xh: float, Vh: float, Xp: float, Vp: float) -> float:
    """
    Intelligent Driver Model (IDM) function
    Xh: Current Vehicle Pos
    Vh: Current Vehicle Velocity  
    Xp: Preceding Vehicle Pos (or stop line)
    Vp: Preceding Vehicle Velocity (or 0)
    
    MATLAB code reference
    """
    # IDM parameters (aggressive settings for urban driving)
    T = 1.3    # Desired Time Gap (s)
    Vd = 17    # Desired Speed (m/s)
    S0 = 0.5   # Min spacing (m)
    a = 2.5    # Max accel (m/s^2)
    b = 2.5    # Comfortable decel (m/s^2)
    L = 1.5    # Vehicle Length (m)
    
    # Gap calculation
    DXh = Xp - Xh - L
    
    # Ensure min gap
    if DXh < S0:
        DXh = S0
    
    # Desired gap
    if (a * b) > 0:
        Rd = S0 + Vh * T + (Vh * (Vh - Vp)) / (2 * (a * b) ** 0.5)
    else:
        Rd = S0 + Vh * T
    
    # IDM acceleration
    if Vd > 0 and DXh > 0:
        speed_term = (Vh / Vd) ** 4
        distance_term = (Rd / DXh) ** 2
        acc = a * (1 - speed_term - distance_term)
        
        # print(f"[IDM DEBUG] Xh={Xh}, Vh={Vh:.1f}, Xp={Xp}, Vp={Vp}")
        # print(f"[IDM DEBUG] DXh={DXh:.1f}, Rd={Rd:.1f}, speed_term={speed_term:.3f}, distance_term={distance_term:.3f}")
        # print(f"[IDM DEBUG] Raw acceleration: {acc:.2f} m/s²")
    else:
        acc = -b  # Emergency brake
        # print(f"[IDM DEBUG] Emergency brake: Vd={Vd}, DXh={DXh}, acc={acc}")
    
    # Clip acceleration
    if acc < -5:
        acc = -5
    elif acc > 4:
        acc = 4
    
    # print(f"[IDM DEBUG] Final acceleration: {acc:.2f} m/s²")
    return acc
