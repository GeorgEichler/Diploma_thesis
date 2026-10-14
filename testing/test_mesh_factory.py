"""Check base-mesh geometry, conformity, and integration with both FEM runners."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from skfem import MeshTri

from afem.core.afem import run_afem
from afem.core.config import AFEMConfig
from afem.core.mesh_factory import make_mesh
from afem.core.ufem import run_ufem
from afem.problems.rhs import constant_one


class MeshFactoryTests(unittest.TestCase):
    def test_four_triangles_per_square_and_conforming_interfaces(self):
        mesh = make_mesh("lshape", initial_mesh="lshape_four_triangles")
        self.assertEqual(mesh.nelements, 12)
        self.assertEqual(mesh.nvertices, 11)
        vertices = mesh.p[:, mesh.t]
        edge_a = vertices[:, 1] - vertices[:, 0]
        edge_b = vertices[:, 2] - vertices[:, 0]
        areas = 0.5 * np.abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
        np.testing.assert_allclose(areas, 0.25)
        self.assertAlmostEqual(float(areas.sum()), 3)
        centroids = vertices.mean(axis=1)
        for x_positive, y_positive in ((False, False), (False, True), (True, False)):
            inside = ((centroids[0] > 0) == x_positive) & ((centroids[1] > 0) == y_positive)
            self.assertEqual(int(inside.sum()), 4)
        self.assertFalse(np.any((centroids[0] > 0) & (centroids[1] > 0)))
        boundary_edges = mesh.p[:, mesh.facets[:, mesh.boundary_facets()]]
        self.assertEqual(boundary_edges.shape[2], 8)
        np.testing.assert_allclose(np.linalg.norm(boundary_edges[:, 1] - boundary_edges[:, 0], axis=0), 1)
        # Both shared square edges must be interior facets with two neighbours.
        for endpoints in ((3, 4), (1, 4)):
            matches = np.all(mesh.facets == np.array(endpoints)[:, None], axis=0)
            self.assertEqual(int(matches.sum()), 1)
            self.assertTrue(np.all(mesh.f2t[:, matches] >= 0))

    def test_initial_refinements_apply_to_selected_base_mesh(self):
        for levels in range(3):
            with self.subTest(levels=levels):
                mesh = make_mesh("lshape", levels, initial_mesh="lshape_four_triangles")
                self.assertEqual(mesh.nelements, 12 * 4**levels)

    def test_default_mesh_preserves_existing_triangulation(self):
        self.assertEqual(AFEMConfig().initial_mesh, "default")
        expected = MeshTri.init_lshaped().refined(1)
        actual = make_mesh("lshape", 1)
        np.testing.assert_array_equal(actual.p, expected.p)
        np.testing.assert_array_equal(actual.t, expected.t)

    def test_invalid_mesh_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown initial_mesh"):
            make_mesh("lshape", initial_mesh="unknown")
        for domain in ("unit_square", "unit_cube"):
            with self.subTest(domain=domain), self.assertRaisesRegex(ValueError, "requires domain"):
                make_mesh(domain, initial_mesh="lshape_four_triangles")

    def test_both_runners_use_selected_mesh_and_save_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AFEMConfig(
                domain="lshape", initial_mesh="lshape_four_triangles",
                initial_refinements=0, max_iterations=1,
                save_plots=False, save_history_plots=False,
                output_dir=Path(tmp),
            )
            solutions = []
            for name, runner in (("adaptive", run_afem), ("uniform", run_ufem)):
                cfg = replace(config, output_dir=Path(tmp) / name)
                mesh, u, history = runner(cfg, constant_one)
                self.assertEqual(history[0]["nelems"], 12)
                self.assertEqual(history[0]["ndofs"], 11)
                self.assertTrue(np.all(np.isfinite(u)))
                self.assertGreater(float(np.max(np.abs(u))), 0)
                saved = json.loads((cfg.output_dir / "history.json").read_text())
                self.assertEqual(saved["config"]["initial_mesh"], "lshape_four_triangles")
                solutions.append(u)
            np.testing.assert_allclose(*solutions)
            _, _, history = run_ufem(replace(config, max_iterations=2), constant_one)
            self.assertEqual([h["nelems"] for h in history], [12, 48])


if __name__ == "__main__":
    unittest.main()
