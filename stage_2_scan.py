#!/usr/bin/env python

# This script runs coil optimizations, one after another, choosing the weights
# and target values from a random distribution. This is effectively a crude form
# of global optimization.

import os
import json
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    curves_to_vtk,
    create_equally_spaced_curves,
    SurfaceRZFourier,
    LinkingNumber,
    CurveLength,
    CurveCurveDistance,
    MeanSquaredCurvature,
    LpCurveCurvature,
    CurveSurfaceDistance,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty

# File for the target plasma surface. It can be either a wout or vmec input file.
plasma_filename = (
    "/Users/mattland/simsopt/simsopt/tests/test_files/input.LandremanPaul2021_QA_lowres"
)

# Number of unique coil shapes, i.e. the number of coils per half field period.
ncoils = 4

# Resolution on the plasma surface for the B_normal objective:
nphi = 32
ntheta = 34

# Number of iterations to perform in each optimization:
MAXITER = 500

#######################################################
# End of input parameters for now. A bunch of other parameters related to the
# distribution of weights and target values can be found at the end of this
# script, in the loop where the optimizations are launched.
#######################################################

# # Threshold and weight for the coil-to-surface distance penalty in the objective function:
# CS_THRESHOLD = 0.3
# CS_WEIGHT = 10

# Load the target plasma surface:
basename = os.path.basename(plasma_filename)
if basename[:4] == "wout":
    surf = SurfaceRZFourier.from_wout(
        plasma_filename, range="half period", nphi=nphi, ntheta=ntheta
    )
else:
    surf = SurfaceRZFourier.from_vmec_input(
        plasma_filename, range="half period", nphi=nphi, ntheta=ntheta
    )

nfp = surf.nfp
R0 = surf.get_rc(0, 0)

# Create a copy of the surface that is closed in theta and phi, and covers the
# full torus toroidally. This is nice for visualization.
nphi_big = nphi * 2 * nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)
surf_big = SurfaceRZFourier(
    dofs=surf.dofs,
    nfp=nfp,
    mpol=surf.mpol,
    ntor=surf.ntor,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta,
)


def run_optimization(
    R1,
    order,
    length_target,
    length_weight,
    max_curvature_threshold,
    max_curvature_weight,
    msc_threshold,
    msc_weight,
    cc_threshold,
    cc_weight,
    index,
):
    directory = (
        f"ncoils_{ncoils}_order_{order}_R1_{R1:.2}_length_target_{length_target:.2}_weight_{length_weight:.2}"
        + f"_max_curvature_{max_curvature_threshold:.2}_weight_{max_curvature_weight:.2}"
        + f"_msc_{msc_threshold:.2}_weight_{msc_weight:.2}"
        + f"_cc_{cc_threshold:.2}_weight_{cc_weight:.2}"
    )

    print()
    print("***********************************************")
    print(f"Job {index}")
    print("Parameters:", directory)
    print("***********************************************")
    print()

    # Directory for output
    OUT_DIR = directory + "/"
    os.mkdir(directory)

    # Create the initial coils:
    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=True,
        R0=R0,
        R1=R1,
        order=order,
        numquadpoints=order * 16,
    )
    # base_currents = [Current(1e5) for i in range(ncoils)]
    base_currents = [Current(1.0) * (1e5) for i in range(ncoils)]
    # Since the target field is zero, one possible solution is just to set all
    # currents to 0. To avoid the minimizer finding that solution, we fix one
    # of the currents:
    base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    curves = [c.curve for c in coils]
    curves_to_vtk(curves, OUT_DIR + "curves_init", close=True)
    pointData = {
        "B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)[
            :, :, None
        ]
    }
    surf.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

    surf_big.to_vtk(OUT_DIR + "surf_big")

    # Define the individual terms objective function:
    Jf = SquaredFlux(surf, bs, definition="local")
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    # Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, max_curvature_threshold) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    JF = (
        Jf
        + length_weight * QuadraticPenalty(sum(Jls), length_target * ncoils)
        + cc_weight * Jccdist
        # + CS_WEIGHT * Jcsdist
        + max_curvature_weight * sum(Jcs)
        + msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs)
        + LinkingNumber(curves, 2)
    )

    iteration = 0

    def fun(dofs):
        nonlocal iteration
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(
            np.abs(
                np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)
            )
        )
        outstr = f"{iteration:4}  J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        # outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        iteration += 1
        return J, grad

    # print("""
    # ################################################################################
    # ### Perform a Taylor test ######################################################
    # ################################################################################
    # """)
    # f = fun
    # dofs = JF.x
    # np.random.seed(1)
    # h = np.random.uniform(size=dofs.shape)
    # J0, dJ0 = f(dofs)
    # dJh = sum(dJ0 * h)
    # for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    #     J1, _ = f(dofs + eps*h)
    #     J2, _ = f(dofs - eps*h)
    #     print("err", (J1-J2)/(2*eps) - dJh)

    # print("""
    # ################################################################################
    # ### Run the optimisation #######################################################
    # ################################################################################
    # """)
    res = minimize(
        fun,
        JF.x,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAXITER, "maxcor": 300},
        tol=1e-15,
    )
    JF.x = res.x
    print(res.message)
    curves_to_vtk(curves, OUT_DIR + "curves_opt", close=True)

    pointData = {
        "B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)[
            :, :, None
        ]
    }
    surf.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)

    bs_big = BiotSavart(coils)
    bs_big.set_points(surf_big.gamma().reshape((-1, 3)))
    pointData = {
        "B_N": np.sum(
            bs_big.B().reshape((nphi_big, ntheta_big, 3)) * surf_big.unitnormal(),
            axis=2,
        )[:, :, None]
    }
    surf_big.to_vtk(OUT_DIR + "surf_big_opt", extra_data=pointData)

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    bs.save(OUT_DIR + "biot_savart.json")

    BdotN = np.mean(
        np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2))
    )

    results = {
        "nfp": nfp,
        "ncoils": ncoils,
        "R0": R0,
        "R1": R1,
        "ncoils": ncoils,
        "order": order,
        "nphi": nphi,
        "ntheta": ntheta,
        "length_target": length_target,
        "length_weight": length_weight,
        "max_curvature_threshold": max_curvature_threshold,
        "max_curvature_weight": max_curvature_weight,
        "msc_threshold": msc_threshold,
        "msc_weight": msc_weight,
        "JF": float(JF.J()),
        "Jf": float(Jf.J()),
        "BdotN": BdotN,
        "lengths": [float(J.J()) for J in Jls],
        "length": float(sum(J.J() for J in Jls)),
        "max_curvatures": [np.max(c.kappa()) for c in base_curves],
        "max_max_curvature": max(np.max(c.kappa()) for c in base_curves),
        "coil_coil_distance": Jccdist.shortest_distance(),
        "cc_threshold": cc_threshold,
        "cc_weight": cc_weight,
        "gradient_norm": np.linalg.norm(JF.dJ()),
        "linking_number": LinkingNumber(curves).J(),
        "directory": directory,
        "mean_squared_curvatures": [float(J.J()) for J in Jmscs],
        "max_mean_squared_curvature": float(max(J.J() for J in Jmscs)),
        "message": res.message,
        "success": res.success,
        "iterations": res.nit,
        "function_evaluations": res.nfev,
        "coil_currents": [c.get_value() for c in base_currents],
    }

    with open(OUT_DIR + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)


#########################################################################
# Carry out the scan. Below you can adjust the ranges for the random weights and
# thresholds.
#########################################################################


def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min


for index in range(10000):
    # Initial radius of the coils:
    R1 = np.random.rand() * 0.3 + 0.45

    # Number of Fourier modes describing each Cartesian component of each coil:
    order = int(np.round(rand(4, 16)))

    # Target length (per coil!) and weight for the length term in the objective function:
    length_target = rand(5, 7)
    length_weight = 10.0 ** rand(-1, 1)

    # Threshold and weight for the curvature penalty in the objective function:
    max_curvature_threshold = rand(5, 15)
    max_curvature_weight = 10.0 ** rand(-7, -4)

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    msc_threshold = rand(5, 15)
    msc_weight = 10.0 ** rand(-7, -4)

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    cc_threshold = rand(0.05, 0.12)
    cc_weight = 10.0 ** rand(-1, 4)

    run_optimization(
        R1,
        order,
        length_target,
        length_weight,
        max_curvature_threshold,
        max_curvature_weight,
        msc_threshold,
        msc_weight,
        cc_threshold,
        cc_weight,
        index,
    )
