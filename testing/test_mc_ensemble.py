"""Run with python -m unittest testing.test_mc_ensemble."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np

from afem.problems.rhs import manufactured_sine_2d
from afem.utils.ensemble import METRICS, aggregate_histories
from experiments import run_mc_ensemble as experiment


def entry(ndofs, value):
    return {"ndofs": ndofs, **{key: value for key, *_ in METRICS}}


class EnsembleTests(unittest.TestCase):
    def test_alignment_common_range_and_statistics(self):
        a = [entry(10, 99), entry(10, 1), entry(100, 3)]
        b = [entry(10, 3), entry(1000, 7)]
        stats = aggregate_histories([a, b], grid_points=3)
        np.testing.assert_allclose(stats["ndofs"], [10, np.sqrt(1000), 100])
        for metric in stats["metrics"].values():
            np.testing.assert_allclose(metric["mean"], [2, 3, 4])
            np.testing.assert_allclose(metric["median"], [2, 3, 4])
            np.testing.assert_allclose(metric["q25"], [1.5, 2.5, 3.5])
            np.testing.assert_allclose(metric["q75"], [2.5, 3.5, 4.5])

    def test_single_point_zeros_and_invalid_values(self):
        stats = aggregate_histories([[entry(10, 0), entry(10, 0)], [entry(10, 0), entry(100, 1)]])
        self.assertEqual(stats["ndofs"], [10])
        self.assertEqual(stats["metrics"]["estimator"]["mean"], [0])
        for histories in ([], [[]], [[entry(10, float("nan"))]],
                          [[entry(10, 1)], [entry(20, 2)]]):
            with self.assertRaises(ValueError):
                aggregate_histories(histories)

    def test_runs_saved_sequentially_without_images_and_replot(self):
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
                    manifest = json.loads((out / "ensemble.json").read_text())
                    self.assertEqual(len(manifest["completed_runs"]), len(calls))
                calls.append(config)
                return original(config, rhs)

            with patch.object(experiment, "run_afem", side_effect=checked_run):
                pngs = experiment.run_ensemble(cfg, manufactured_sine_2d, num_runs=2)
            self.assertEqual([p.name for p in pngs], ["ensemble_errors.png", "ensemble_estimator.png"])
            self.assertTrue(all(p.is_file() for p in pngs))
            self.assertEqual([c.mc_seed for c in calls], [cfg.mc_seed, cfg.mc_seed + 2])
            for config in calls:
                self.assertEqual(list(config.output_dir.rglob("*.png")), [])
                self.assertFalse(config.save_plots)
                self.assertFalse(config.save_history_plots)
            first = json.loads((calls[0].output_dir / "history.json").read_text())
            second = json.loads((calls[1].output_dir / "history.json").read_text())
            self.assertNotEqual(first["history"][0]["estimator"], second["history"][0]["estimator"])

            # An old directory must not silently enter a subsequent aggregate.
            stale = out / "run_999"
            stale.mkdir()
            (stale / "history.json").write_text("not a valid run")
            with patch.object(experiment, "run_afem", side_effect=AssertionError("Unexpected solve")):
                for summary, individuals, band in (("mean", False, True), ("none", True, False)):
                    experiment.plot_saved_ensemble(out, summary=summary,
                                                   show_individual=individuals, show_band=band)
            self.assertEqual(json.loads((out / "ensemble_statistics.json").read_text())["nruns"], 2)
            self.assertEqual(list(out.rglob("*.pdf")), [])

    def test_optional_final_mesh_and_interrupted_ensemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = replace(experiment.CONFIG, domain="unit_square", initial_refinements=1,
                          max_iterations=1, reference_quadrature_order=10, output_dir=out)
            original = experiment.run_afem

            def fail_second(config, rhs):
                if config.output_dir.name == "run_002":
                    raise RuntimeError("Simulated interruption")
                return original(config, rhs)

            with patch.object(experiment, "run_afem", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    experiment.run_ensemble(cfg, manufactured_sine_2d, num_runs=2, save_final_meshes=True)
            pngs = list((out / "run_001").rglob("*.png"))
            self.assertEqual([p.name for p in pngs], ["final_mesh.png"])
            experiment.plot_saved_ensemble(out)
            stats = json.loads((out / "ensemble_statistics.json").read_text())
            self.assertEqual(stats["nruns"], 1)


if __name__ == "__main__":
    unittest.main()
