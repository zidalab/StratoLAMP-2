"""
Droplet Digital PCR Quantification Module

This module provides functions for:
- Newton-Raphson iteration for concentration calculation
- Droplet volume estimation from area measurements
- Target concentration quantification from droplet classification
"""

import math
import pandas as pd
import openpyxl as op
import time

def func(pos_volumes, neg_volumes, concentration):
    """
    Function for Newton-Raphson iteration to solve concentration.

    Args:
        pos_volumes (list): Volumes of positive droplets (μL)
        neg_volumes (list): Volumes of negative droplets (μL)
        concentration (float): Current concentration estimate (copies/μL)

    Returns:
        float: Function value at current concentration
    """
    total_volume = sum(pos_volumes) + sum(neg_volumes)
    sigma = 0.0

    for vol in pos_volumes:
            sigma += vol / (1 - math.exp(-vol * concentration))

    return total_volume - sigma


def func_prime(pos_volumes, neg_volumes, concentration):
    """
    Derivative of the Newton-Raphson function.

    Args:
        pos_volumes (list): Volumes of positive droplets (μL)
        neg_volumes (list): Volumes of negative droplets (μL)
        concentration (float): Current concentration estimate (copies/μL)

    Returns:
        float: Derivative value at current concentration
    """
    sigma = 0.0
    for vol in pos_volumes:
        numerator = (vol ** 2) * math.exp(-vol * concentration)
        denominator = (1 - math.exp(-vol * concentration)) ** 2
        sigma += numerator / denominator
    return sigma


def get_concentration(pos_volumes, neg_volumes):
    """
    Calculate concentration using Newton-Raphson iteration.

    Args:
        pos_volumes (list): Volumes of positive droplets (μL)
        neg_volumes (list): Volumes of negative droplets (μL)

    Returns:
        float: Calculated concentration (copies/μL)
    """
    # Initial concentration estimate
    pos_count = float(len(pos_volumes))
    neg_count = float(len(neg_volumes))
    total_count = pos_count + neg_count
    avg_volume = (sum(pos_volumes) + sum(neg_volumes)) / total_count

    concentration = -math.log(1 - pos_count / total_count) / avg_volume

    # Newton-Raphson iteration
    MAX_ITERATIONS = 1000
    TOLERANCE = 1e-9

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}: c = {concentration:.4f}")
        new_concentration = concentration - (
                func(pos_volumes, neg_volumes, concentration) /
                func_prime(pos_volumes, neg_volumes, concentration)
        )

        if abs((new_concentration - concentration) / concentration) < TOLERANCE:
            break

        concentration = new_concentration

    print(f"Final concentration: {concentration} copies/μL")
    return concentration


def calculate_volume(area_px, scale_bar, critical_diameter):
    """
    Calculate droplet volume from area measurement.

    Args:
        area_px (float): Droplet area in pixels
        scale_bar (float): Micrometers per pixel conversion factor
        critical_diameter (float): Threshold diameter for volume calculation mode

    Returns:
        float: Calculated volume (μL)
    """
    area_um2 = area_px * (scale_bar ** 2)
    diameter = math.sqrt(4 * area_um2 / math.pi)

    if diameter > critical_diameter:
        # Chip height is ~40μm, convert to μL (1μm³ = 1e-9μL)
        return area_um2 * 40 * 1e-9
    else:
        # Polynomial fit for small droplets
        return 3.29211e-10 * (diameter ** 3)

def main():
    """Simulate droplet quantification process with generated test data."""
    # Simulation parameters
    NUM_DROPLETS = 10000  # Total number of droplets to simulate
    SCALE_BAR = 1.72  # um/px
    D_CR = 95  # Critical diameter (um)
    SAVE_PATH = "simulated_quantification.xlsx"

    # Generate simulated droplet areas (in pixels)
    np.random.seed(42)  # For reproducible results
    areas = np.random.lognormal(mean=5, sigma=0.5, size=NUM_DROPLETS)

    # Randomly assign classes (0=neg, 1=low, 2=medium, 3=large)
    classes = np.random.choice([0, 1, 2, 3], size=NUM_DROPLETS, p=[0.3, 0.25, 0.25, 0.2])

    # Calculate volumes for each class
    vols_neg = [calculate_volume(a, SCALE_BAR, D_CR) for a, c in zip(areas, classes) if c == 0]
    vols_low = [calculate_volume(a, SCALE_BAR, D_CR) for a, c in zip(areas, classes) if c == 1]
    vols_medium = [calculate_volume(a, SCALE_BAR, D_CR) for a, c in zip(areas, classes) if c == 2]
    vols_large = [calculate_volume(a, SCALE_BAR, D_CR) for a, c in zip(areas, classes) if c == 3]

    # Calculate statistics
    N_total = len(vols_neg) + len(vols_low) + len(vols_medium) + len(vols_large)
    avg_volume = sum(vols_neg + vols_low + vols_medium + vols_large) / N_total
    avg_radius_eq = ((3 * avg_volume) / (4 * math.pi)) ** (1 / 3)

    # Target 1 quantification (low + large vs neg + medium)
    print("\n--- Target 1 Quantification ---")
    vols_pos_t1 = vols_low + vols_large
    vols_neg_t1 = vols_neg + vols_medium

    start = time.time()
    C_target1 = get_concentration(vols_pos_t1, vols_neg_t1)
    elapsed = time.time() - start
    print(f"Calculation time: {elapsed:.6f} seconds")

    # Target 2 quantification (medium + large vs neg + low)
    print("\n--- Target 2 Quantification ---")
    vols_pos_t2 = vols_medium + vols_large
    vols_neg_t2 = vols_neg + vols_low

    start = time.time()
    C_target2 = get_concentration(vols_pos_t2, vols_neg_t2)
    elapsed = time.time() - start
    print(f"Calculation time: {elapsed:.6f} seconds")

    # Save results
    workbook = op.Workbook()
    sheet = workbook.active
    sheet.append(["Parameter", "Target 1", "Target 2"])
    sheet.append(["Concentration (copies/μL)", C_target1, C_target2])
    sheet.append(["Positive Droplets", len(vols_pos_t1), len(vols_pos_t2)])
    sheet.append(["Negative Droplets", len(vols_neg_t1), len(vols_neg_t2)])
    sheet.append(["Total Droplets", N_total, N_total])
    sheet.append(["Avg Radius (μm)", avg_radius_eq, avg_radius_eq])
    workbook.save(SAVE_PATH)


if __name__ == "__main__":
    import numpy as np  # Required for simulation
    main()