"""Run with python -m unittest testing.test_uniform_refinement."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np

from afem.core.afem import run_afem
from afem.core.config import AFEMConfig
from afem.core.mesh_factory import make_mesh
from afem.core.ufem import run_ufem
from afem.problems.rhs import manufactured_sine_2d


class UniformRefinementTests(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.out = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        # These tests exercise numerical results and persistence, not rendering.
        for name in ("plot_history", "plot_reference_error_history"):
            self.resources.enter_context(patch(f"afem.core.afem.{name}"))
        self.config = AFEMConfig(
            initial_refinements=1, max_iterations=3, save_plots=False,
            compute_reference_error=True, output_dir=self.out,
        )

    def test_all_load_methods_refine_every_triangle_and_save_direct_errors(self):
        expected_mesh = make_mesh("unit_square", 3)
        for load, rule in (("quadrature", "midpoint"),
                           ("monte_carlo", "default"),
                           ("quadrature", "default")):
            with self.subTest(load=load, rule=rule):
                config = replace(self.config, load_method=load, quadrature_rule=rule)
                mesh, u, history = run_ufem(config, manufactured_sine_2d)
                self.assertEqual([h["nelems"] for h in history], [8, 32, 128])
                self.assertTrue(all(h["nmarked"] == h["nelems"] for h in history))
                np.testing.assert_allclose(mesh.p, expected_mesh.p)
                np.testing.assert_array_equal(mesh.t, expected_mesh.t)
                self.assertEqual(u.size, history[-1]["ndofs"])
                for key in ("relative_l2_error_ref", "relative_h1_semi_error_ref"):
                    self.assertTrue(all(np.isfinite(h[key]) for h in history))
                    self.assertLess(history[-1][key], history[0][key])
                saved = json.loads((self.out / "history.json").read_text())
                self.assertEqual(saved["config"]["refinement_strategy"], "uniform")
                self.assertEqual(saved["history"], history)
                self.assertEqual(config.refinement_strategy, "adaptive")

    def test_zero_estimator_does_not_stop_uniform_refinement(self):
        rhs = lambda x: np.zeros_like(x[0])
        config = replace(self.config, compute_reference_error=False)
        _, _, uniform = run_ufem(replace(config, theta=0), rhs)
        _, _, adaptive = run_afem(config, rhs)
        self.assertEqual([h["nelems"] for h in uniform], [8, 32, 128])
        self.assertTrue(all(h["estimator"] == 0 for h in uniform))
        self.assertEqual([h["nelems"] for h in adaptive], [8, 8, 8])
        self.assertTrue(all(h["nmarked"] == 0 for h in adaptive))

    def test_energy_errors_and_configuration_entry_point(self):
        config = replace(self.config, max_iterations=2, refinement_strategy="uniform")
        _, _, direct = run_afem(config, manufactured_sine_2d)
        reference = json.loads((self.out / "history.json").read_text())["reference"]
        _, _, energy = run_ufem(
            replace(config, reference_error_method="energy"), manufactured_sine_2d,
        )
        for d, e in zip(direct, energy):
            self.assertIn("energy_reference_mesh", e)
            np.testing.assert_allclose(
                d["relative_h1_semi_error_ref"] * reference["reference_h1_semi_norm"],
                e["h1_semi_error_ref"], rtol=1e-10,
            )

    def test_one_iteration_has_no_extra_refinement(self):
        mesh, _, history = run_ufem(
            replace(self.config, max_iterations=1, compute_reference_error=False),
            manufactured_sine_2d,
        )
        self.assertEqual(mesh.t.shape[1], 8)
        self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
