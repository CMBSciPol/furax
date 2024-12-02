# Necessary imports
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# Healpy and PySM3 imports
import healpy as hp
import pysm3

# FGBuster imports
from fgbuster import (
    CMB,
    Dust,
    Synchrotron,
    MixingMatrix,
    get_observation,
    get_instrument,
)
from fgbuster.visualization import corner_norm
from fgbuster.algebra import _build_bound_inv_logL_and_logL_dB
from fgbuster import (
    CMB,
    Dust,
    Synchrotron,
    basic_comp_sep,
    MixingMatrix,
    get_observation,
    get_instrument,
)

# Furax imports
import jax
import jaxopt
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from furax._base.blocks import BlockDiagonalOperator, BlockRowOperator
from furax._base.core import HomothetyOperator, IdentityOperator
from furax.landscapes import StokesPyTree, HealpixLandscape
from furax.tree import as_structure
from furax.operators.sed import CMBOperator, DustOperator, SynchrotronOperator
import operator
import optax
from furax.optimizers import newton_cg, optimise


def generate_maps(instrument, nside, cache_dir="freq_maps_cache"):
    """Generate or load cached frequency maps."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"freq_maps_nside_{nside}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            freq_maps = pickle.load(f)
        print(f"Loaded freq_maps for nside {nside} from cache.")
    else:
        freq_maps = get_observation(instrument, "c1d0s0", nside=nside)
        with open(cache_file, "wb") as f:
            pickle.dump(freq_maps, f)
        print(f"Generated and saved freq_maps for nside {nside}.")

    print("freq_maps shape:", freq_maps.shape)
    return freq_maps


def run_fgbuster_logL(nside, freq_maps, components, nu):
    """Run FGBuster log-likelihood."""
    print(f"Running FGBuster Log Likelihood with nside={nside} ...")

    A = MixingMatrix(*components)
    A_ev = A.evaluator(nu)
    A_dB_ev = A.diff_evaluator(nu)
    data = freq_maps.T

    logL, _, _ = _build_bound_inv_logL_and_logL_dB(
        A_ev, data, None, A_dB_ev, A.comp_of_dB
    )
    x0 = np.array([x for c in components for x in c.defaults])

    durations = []
    result = logL(x0)
    for _ in range(10):
        start_time = perf_counter()
        logL(x0)
        durations.append(perf_counter() - start_time)

    durations = np.array(durations) * 1000

    return result, durations.mean(), durations.min()


def run_jax_negative_log_prob(
    nside, freq_maps, best_params, nu, structure, dust_nu0, synchrotron_nu0
):
    """Run JAX-based negative log-likelihood."""
    print(f"Running Furax Log Likelihood nside={nside} ...")
    d = StokesPyTree.from_stokes(
        I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :]
    )
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    def make_mixing_matrix_operator(params, nu, in_structure):
        cmb = CMBOperator(nu, in_structure=in_structure)
        dust = DustOperator(
            nu,
            frequency0=dust_nu0,
            temperature=params["temp_dust"],
            beta=params["beta_dust"],
            in_structure=in_structure,
        )
        synchrotron = SynchrotronOperator(
            nu,
            frequency0=synchrotron_nu0,
            beta_pl=params["beta_pl"],
            in_structure=in_structure,
        )
        sed = BlockDiagonalOperator(
            {"cmb": cmb, "dust": dust, "synchrotron": synchrotron}
        )
        integ = BlockRowOperator(
            {
                component: IdentityOperator(sed.blocks[component].out_structure())
                for component in sed.blocks
            }
        )
        return (integ @ sed).reduce()

    @jax.jit
    def negative_log_prob(params, d):
        A = make_mixing_matrix_operator(params, nu, in_structure=structure)
        x = (A.T @ invN)(d)
        l = jax.tree.map(lambda a, b: a @ b, x, (A.T @ invN @ A).I(x))
        return -jax.tree.reduce(operator.add, l)

    result = negative_log_prob(best_params, d).block_until_ready()

    durations = []
    for _ in range(10):
        start_time = perf_counter()
        negative_log_prob(best_params, d).block_until_ready()
        durations.append(perf_counter() - start_time)

    durations = np.array(durations) * 1000
    return result, durations.mean(), durations.min()


def run_jax_lbfgs(nside , freq_maps, best_params, nu, structure, dust_nu0, synchrotron_nu0):
    """Run JAX-based negative log-likelihood."""

    print(f"Running Furax LBGS Comp sep nside={nside} ...")

    d = StokesPyTree.from_stokes(
        I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :]
    )
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    def make_mixing_matrix_operator(params, nu, in_structure):
        cmb = CMBOperator(nu, in_structure=in_structure)
        dust = DustOperator(
            nu,
            frequency0=dust_nu0,
            temperature=params["temp_dust"],
            beta=params["beta_dust"],
            in_structure=in_structure,
        )
        synchrotron = SynchrotronOperator(
            nu,
            frequency0=synchrotron_nu0,
            beta_pl=params["beta_pl"],
            in_structure=in_structure,
        )
        sed = BlockDiagonalOperator(
            {"cmb": cmb, "dust": dust, "synchrotron": synchrotron}
        )
        integ = BlockRowOperator(
            {
                component: IdentityOperator(sed.blocks[component].out_structure())
                for component in sed.blocks
            }
        )
        return (integ @ sed).reduce()

    @jax.jit
    def negative_log_prob(params, d):
        A = make_mixing_matrix_operator(params, nu, in_structure=structure)
        x = (A.T @ invN)(d)
        l = jax.tree.map(lambda a, b: a @ b, x, (A.T @ invN @ A).I(x))
        return -jax.tree.reduce(operator.add, l)

    solver = optax.lbfgs()

    final_params, final_state = optimise(
        best_params,
        negative_log_prob,
        solver,
        max_iter=100,
        tol=1e-5,
        d=d,
    )
    final_params["beta_pl"].block_until_ready()

    durations = []
    for _ in range(10):
        start_time = perf_counter()
        final_params, final_state = optimise(
            best_params,
            negative_log_prob,
            solver,
            max_iter=100,
            tol=1e-5,
            d=d,
        )
        final_params["beta_pl"].block_until_ready()
        durations.append(perf_counter() - start_time)

    durations = np.array(durations) * 1000

    print(f"\tResults : \n\t\t{final_params}")

    return final_params, durations.mean(), durations.min()


def run_jax_tnc(nside , freq_maps, best_params, nu, structure, dust_nu0, synchrotron_nu0):
    """Run JAX-based negative log-likelihood."""

    print(f"Running Furax TNC From SciPy Comp sep nside={nside} ...")

    d = StokesPyTree.from_stokes(
        I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :]
    )
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    def make_mixing_matrix_operator(params, nu, in_structure):
        cmb = CMBOperator(nu, in_structure=in_structure)
        dust = DustOperator(
            nu,
            frequency0=dust_nu0,
            temperature=params["temp_dust"],
            beta=params["beta_dust"],
            in_structure=in_structure,
        )
        synchrotron = SynchrotronOperator(
            nu,
            frequency0=synchrotron_nu0,
            beta_pl=params["beta_pl"],
            in_structure=in_structure,
        )
        sed = BlockDiagonalOperator(
            {"cmb": cmb, "dust": dust, "synchrotron": synchrotron}
        )
        integ = BlockRowOperator(
            {
                component: IdentityOperator(sed.blocks[component].out_structure())
                for component in sed.blocks
            }
        )
        return (integ @ sed).reduce()

    @jax.jit
    def negative_log_prob(params, d):
        A = make_mixing_matrix_operator(params, nu, in_structure=structure)
        x = (A.T @ invN)(d)
        l = jax.tree.map(lambda a, b: a @ b, x, (A.T @ invN @ A).I(x))
        return -jax.tree.reduce(operator.add, l)

    scipy_solver = jaxopt.ScipyMinimize(
        fun=negative_log_prob, method="TNC", jit=True, tol=1e-6
    )
    result = scipy_solver.run(best_params, d)
    result.params["beta_pl"].block_until_ready()

    durations = []
    for _ in range(10):
        start_time = perf_counter()
        result = scipy_solver.run(best_params, d)
        result.params["beta_pl"].block_until_ready()
        durations.append(perf_counter() - start_time)

    durations = np.array(durations) * 1000
    print(f"\tResults : \n\t\t{result.params}")
    return result.params, durations.mean(), durations.min()


def run_fgbuster_comp_sep(nside, instrument, best_params, freq_maps, components, nu):
    """Run FGBuster log-likelihood."""
    print(f"Running FGBuster Comp sep nside={nside} ...")

    components[1]._set_default_of_free_symbols(
        beta_d=best_params["beta_dust"], temp=best_params["temp_dust"]
    )
    components[2]._set_default_of_free_symbols(beta_pl=best_params["beta_pl"])

    result = basic_comp_sep(components, instrument, freq_maps)

    durations = []
    for _ in range(10):
        start_time = perf_counter()
        result = basic_comp_sep(components, instrument, freq_maps)
        durations.append(perf_counter() - start_time)

    durations = np.array(durations) * 1000
    print(f"\tResults : \n\t\t{result.params}\n\t\t{result.x}")

    return result, durations.mean(), durations.min()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark FGBuster and Furax Component Separation Methods"
    )
    parser.add_argument(
        "-n",
        "--nsides",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="List of nsides to benchmark",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="comparison",
        help="Output filename prefix for the plots",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(10, 6),
        help="Figure size for the plots (width, height)",
    )
    parser.add_argument(
        "-l",
        "--likelihood",
        action="store_true",
        help="Benchmark FGBuster and Furax log-likelihood methods",
    )
    parser.add_argument(
        "-s",
        "--solvers",
        action="store_true",
        help="Benchmark solvers: FGBuster, JAX LBFGS, and JAX TNC",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    instrument = get_instrument("LiteBIRD")
    nu = instrument["frequency"].values
    cache_dir = "freq_maps_cache"
    stokes_type = "IQU"
    dust_nu0, synchrotron_nu0 = 150.0, 20.0
    components = [CMB(), Dust(dust_nu0), Synchrotron(synchrotron_nu0)]
    best_params = {"temp_dust": 20.0, "beta_dust": 1.54, "beta_pl": -3.0}

    fgbuster_likelihood_times, furax_likelihood_times = [], []
    fgbuster_solver_times, lbfgs_solver_times, tnc_solver_times = [], [], []
    nside_values = []

    for nside in args.nsides:
        freq_maps = generate_maps(instrument, nside, cache_dir)
        structure = HealpixLandscape(nside, stokes_type).structure

        if args.likelihood:
            # Likelihood mode benchmarking
            print(f"Running likelihood benchmarking for nside={nside}...")
            _, fgbuster_mean_time, _ = run_fgbuster_logL(nside, freq_maps, components, nu)
            _, furax_mean_time, _ = run_jax_negative_log_prob(
                nside, freq_maps, best_params, nu, structure, dust_nu0, synchrotron_nu0
            )
            fgbuster_likelihood_times.append(fgbuster_mean_time)
            furax_likelihood_times.append(furax_mean_time)
            print(f"\tTime taken by FGBuster {fgbuster_mean_time:.4f}")
            print(f"\tTime taken by FURAX {furax_mean_time:.4f}")

        if args.solvers:
            # Solver mode benchmarking
            print(f"Running solver benchmarking for nside={nside}...")
            _, fgbuster_solver_time, _ = run_fgbuster_comp_sep(
                nside, instrument, best_params, freq_maps, components, nu
            )
            _, lbfgs_solver_time, _ = run_jax_lbfgs(
                nside , freq_maps, best_params, nu, structure, dust_nu0, synchrotron_nu0
            )
            _, tnc_solver_time, _ = run_jax_tnc(
                nside , freq_maps, best_params, nu, structure, dust_nu0, synchrotron_nu0
            )
            fgbuster_solver_times.append(fgbuster_solver_time)
            lbfgs_solver_times.append(lbfgs_solver_time)
            tnc_solver_times.append(tnc_solver_time)
            print(f"\tTime taken by FGBuster {fgbuster_solver_time:.4f}")
            print(f"\tTime taken by FURAX TNC {tnc_solver_time:.4f}")
            print(f"\tTime taken by FURAX LBFGS {lbfgs_solver_time:.4f}")

        nside_values.append(nside)

    # Plot log-likelihood results
    if args.likelihood:
        plt.figure(figsize=args.figsize)
        plt.plot(nside_values, fgbuster_likelihood_times, label="Fgbuster Log Likelihood", marker="o")
        plt.plot(nside_values, furax_likelihood_times, label="Furax Log Likelihood", marker="x")
        plt.xlabel("nside")
        plt.ylabel("Time (ms)")
        plt.yscale("log")
        plt.xscale("log", base=2)
        plt.title("Runtime Comparison: Log Likelihood")
        plt.legend()
        likelihood_output = f"{args.output}_log_likelihood.png"
        plt.savefig(likelihood_output, transparent=True)
        print(f"Likelihood plot saved to {likelihood_output}")

    # Plot solver results
    if args.solvers:
        plt.figure(figsize=args.figsize)
        plt.plot(nside_values, fgbuster_solver_times, label="Fgbuster Solver", marker="o")
        plt.plot(nside_values, lbfgs_solver_times, label="Furax LBFGS Solver", marker="x")
        plt.plot(nside_values, tnc_solver_times, label="Furax TNC Solver", marker="^")
        plt.xlabel("nside")
        plt.ylabel("Time (ms)")
        plt.yscale("log")
        plt.xscale("log", base=2)
        plt.title("Runtime Comparison: Solvers")
        plt.legend()
        solvers_output = f"{args.output}_solvers.png"
        plt.savefig(solvers_output, transparent=True)
        print(f"Solvers plot saved to {solvers_output}")


if __name__ == "__main__":
    main()
