"""Run with python -m unittest testing.test_refinement_comparison."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afem.problems.rhs import manufactured_sine_2d
from experiments import run_refinement_comparison as experiment


class RefinementComparisonTests(unittest.TestCase):
    def test_all_methods_sequential_saves_and_plot_only(self):
        with ExitStack() as resources:
            root = Path(resources.enter_context(tempfile.TemporaryDirectory()))
            # Tiny test meshes do not reach the experiment's paper-scale slope anchors.
            resources.enter_context(patch.object(experiment, "REFERENCE_RATES", ()))
            resources.enter_context(patch.object(experiment, "REFERENCE_STARTS", None))
            for name in ("plot_history", "plot_reference_error_history"):
                resources.enter_context(patch(f"afem.core.afem.{name}"))
            cfg = replace(
                experiment.CONFIG, domain="unit_square", initial_mesh="default", initial_refinements=1,
                max_iterations=2, reference_quadrature_order=10,
                refinement_strategy="uniform", save_plots=False, output_dir=root,
            )
            original_uniform = experiment.run_ufem

            for method, (load, rule, _) in experiment.METHODS.items():
                with self.subTest(method=method):
                    def checked_uniform(config, rhs):
                        adaptive_dir = root / method / "adaptive"
                        for filename in ("history.json", "final_solution.npz", "experiment.json"):
                            self.assertTrue((adaptive_dir / filename).is_file())
                        return original_uniform(config, rhs)

                    with patch.object(experiment, "run_ufem", side_effect=checked_uniform) as uniform:
                        pdf = experiment.run_comparison(
                            cfg, manufactured_sine_2d, method=method, uniform_iterations=3,
                        )
                        self.assertEqual(uniform.call_count, 1)

                    self.assertTrue(pdf.is_file())
                    self.assertEqual(pdf.suffix, ".pdf")
                    self.assertEqual(list((root / method).rglob("*.png")), [])
                    for strategy, levels in (("adaptive", 2), ("uniform", 3)):
                        folder = root / method / strategy
                        data = json.loads((folder / "history.json").read_text())
                        self.assertEqual(len(data["history"]), levels)
                        self.assertEqual(data["config"]["refinement_strategy"], strategy)
                        self.assertEqual(data["config"]["load_method"], load)
                        self.assertEqual(data["config"]["quadrature_rule"], rule)
                        self.assertEqual(data["config"]["mc_seed"], cfg.mc_seed)
                        self.assertEqual(data["history"][0]["nelems"], 8)
                        if strategy == "uniform":
                            self.assertEqual([h["nelems"] for h in data["history"]], [8, 32, 128])
                        with np.load(folder / "final_solution.npz") as saved:
                            self.assertEqual(saved["u"].size, data["history"][-1]["ndofs"])

                    with patch.object(experiment, "run_afem", side_effect=AssertionError("Unexpected solve")), \
                         patch.object(experiment, "run_ufem", side_effect=AssertionError("Unexpected solve")), \
                         patch("afem.utils.plotting.plt.close"):
                        experiment.plot_saved_comparison(root, method)
                        ax = plt.gcf().axes[0]
                        self.assertEqual(len(ax.lines), 4)
                        self.assertEqual(ax.get_title(), "")
                        for i, (strategy, _, color) in enumerate(experiment.STRATEGIES):
                            solid, dashed = ax.lines[2*i:2*i+2]
                            self.assertEqual(solid.get_color(), color)
                            self.assertEqual(dashed.get_color(), color)
                            self.assertEqual(solid.get_linestyle(), "-")
                            self.assertEqual(dashed.get_linestyle(), "--")
                            data = json.loads((root / method / strategy / "history.json").read_text())
                            np.testing.assert_allclose(solid.get_ydata(), [h["relative_h1_semi_error_ref"] for h in data["history"]])
                            np.testing.assert_allclose(dashed.get_ydata(), [h["relative_l2_error_ref"] for h in data["history"]])
                    plt.close("all")
                    experiment.plot_saved_comparison(root, method, save_png=True)
                    self.assertTrue(pdf.with_suffix(".png").is_file())

    def test_invalid_choices_fail_before_solving(self):
        with patch.object(experiment, "run_afem") as solve:
            with self.assertRaises(ValueError):
                experiment.run_comparison(experiment.CONFIG, manufactured_sine_2d,
                                         method="unknown", uniform_iterations=2)
            with self.assertRaises(ValueError):
                experiment.run_comparison(experiment.CONFIG, manufactured_sine_2d,
                                         method="quadrature", uniform_iterations=0)
            solve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
