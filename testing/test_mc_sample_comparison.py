"""Run with python -m unittest testing.test_mc_sample_comparison."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afem.problems.rhs import manufactured_sine_2d
from experiments import run_mc_sample_comparison as experiment


class SampleComparisonTests(unittest.TestCase):
    def test_validation_before_solving(self):
        np.testing.assert_array_equal(experiment._validate_counts(np.array([2, 5])), [2, 5])
        for counts in ([], [0], [-1], [1.5], [True], [2, 2]):
            with self.subTest(counts=counts), patch.object(experiment, "run_afem") as solve:
                with self.assertRaises(ValueError):
                    experiment.run_comparison(experiment.CONFIG, manufactured_sine_2d, sample_counts=counts)
                solve.assert_not_called()

    def test_sequential_counts_saved_and_replotted_with_ordered_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = replace(experiment.CONFIG, domain="unit_square", initial_refinements=1,
                          max_iterations=2, reference_quadrature_order=10, output_dir=out)
            original = experiment.run_afem
            calls = []

            def checked_run(config, rhs):
                if calls:
                    previous = calls[-1].output_dir
                    self.assertTrue((previous / "history.json").is_file())
                    self.assertTrue((previous / "final_solution.npz").is_file())
                    manifest = json.loads((out / "sample_comparison.json").read_text())
                    self.assertEqual(len(manifest["completed_runs"]), len(calls))
                calls.append(config)
                return original(config, rhs)

            with patch.object(experiment, "run_afem", side_effect=checked_run):
                paths = experiment.run_comparison(cfg, manufactured_sine_2d, sample_counts=[10, 1, 2])
            self.assertEqual([c.mc_samples_per_element for c in calls], [10, 1, 2])
            used_seeds = []
            for config in calls:
                count = config.mc_samples_per_element
                self.assertEqual(config.mc_seed, cfg.mc_seed + count * cfg.max_iterations)
                used_seeds.extend(range(config.mc_seed, config.mc_seed + cfg.max_iterations))
                data = json.loads((config.output_dir / "history.json").read_text())
                self.assertEqual(data["config"]["mc_samples_per_element"], count)
                self.assertEqual(len(data["history"]), 2)
                self.assertEqual(list(config.output_dir.rglob("*.png")), [])
            self.assertEqual(len(used_seeds), len(set(used_seeds)))
            self.assertTrue(all(p.is_file() for p in paths))
            self.assertEqual([p.name for p in paths], ["sample_errors.png", "sample_estimator.png"])
            self.assertEqual(list(out.rglob("*.pdf")), [])

            stale = out / "samples_99"
            stale.mkdir()
            (stale / "history.json").write_text("stale invalid data")
            with patch.object(experiment, "run_afem", side_effect=AssertionError("Unexpected solve")), \
                 patch("afem.utils.sample_comparison.plt.close"):
                experiment.plot_saved_comparison(out)
                figures = [plt.figure(n) for n in plt.get_fignums()]
                self.assertEqual(len(figures), 2)
                error_ax = figures[0].axes[0]
                residual_ax = figures[1].axes[0]
                self.assertEqual(error_ax.get_ylabel(), "relative error")
                self.assertEqual(len(error_ax.lines), 6)
                self.assertEqual(len(residual_ax.lines), 3)
                brightness = []
                for index, count in enumerate([1, 2, 10]):
                    h1, l2 = error_ax.lines[2*index:2*index+2]
                    self.assertEqual(h1.get_linestyle(), "-")
                    self.assertEqual(l2.get_linestyle(), "--")
                    np.testing.assert_allclose(h1.get_color(), l2.get_color())
                    np.testing.assert_allclose(h1.get_color(), residual_ax.lines[index].get_color())
                    brightness.append(np.mean(h1.get_color()[:3]))
                    data = json.loads((out / f"samples_{count}" / "history.json").read_text())
                    np.testing.assert_allclose(l2.get_ydata(), [h["relative_l2_error_ref"] for h in data["history"]])
                self.assertTrue(all(a > b for a, b in zip(brightness, brightness[1:])))
                for fig in figures:
                    self.assertEqual(fig.axes[0].get_title(), "")
                    self.assertIsNone(fig._suptitle)
                    self.assertEqual([t.get_text() for t in fig.axes[1].get_yticklabels()], ["1", "2", "10"])
                    self.assertEqual(fig.axes[1].get_yscale(), "log")
                    color_scale = fig.axes[1].collections[-1]
                    # Two samples should lie at log10(2), not halfway by rank.
                    self.assertAlmostEqual(float(color_scale.norm(2)), np.log10(2))
                    np.testing.assert_allclose(error_ax.lines[2].get_color(),
                                               color_scale.cmap(color_scale.norm(2)))
            plt.close("all")

    def test_interrupted_study_can_plot_one_completed_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = replace(experiment.CONFIG, domain="unit_square", initial_refinements=1,
                          max_iterations=1, reference_quadrature_order=10, output_dir=Path(tmp))
            original = experiment.run_afem

            def fail_second(config, rhs):
                if config.mc_samples_per_element == 2:
                    raise RuntimeError("Simulated interruption")
                return original(config, rhs)

            with patch.object(experiment, "run_afem", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    experiment.run_comparison(cfg, manufactured_sine_2d, sample_counts=[1, 2])
            paths = experiment.plot_saved_comparison(cfg.output_dir)
            self.assertTrue(all(p.is_file() for p in paths))


if __name__ == "__main__":
    unittest.main()
