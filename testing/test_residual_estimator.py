"""Run with python -m unittest testing.test_residual_estimator."""

import unittest

import numpy as np
from skfem import MeshTri, MeshTet

from afem.core.estimator import residual_estimator
from afem.core.geometry import element_geometry
from afem.problems.rhs import constant_one, high_oscillation


class ResidualEstimatorTests(unittest.TestCase):
    def test_linear_rhs_squared_integral_in_2d_and_3d(self):
        # Analytic simplex moment: integral x^2 = volume *
        # ((sum vertex x)^2 + sum vertex x^2) / ((dim+1)*(dim+2)).
        # This distinguishes integrating f^2 from squaring the mean of f.
        for mesh in (MeshTri(), MeshTet()):
            with self.subTest(dim=mesh.p.shape[0]):
                u = np.zeros(mesh.p.shape[1])
                vols, hs, _, _ = element_geometry(mesh)
                xs = mesh.p[0, mesh.t]
                dim = mesh.p.shape[0]
                integral = vols * (xs.sum(axis=0)**2 + (xs**2).sum(axis=0)) / ((dim+1)*(dim+2))
                eta = residual_estimator(mesh, u, lambda x: x[0], quadrature_order=2)
                np.testing.assert_allclose(eta**2, hs**2 * integral)

    def test_midpoint_still_misses_centroid_zeros(self):
        mesh = MeshTri().refined(3)
        u = np.zeros(mesh.p.shape[1])
        midpoint = residual_estimator(
            mesh, u, high_oscillation, quadrature_rule="midpoint", quadrature_order=10,
        )
        quadrature = residual_estimator(mesh, u, high_oscillation, quadrature_order=10)
        self.assertLess(np.linalg.norm(midpoint), 1e-12)
        self.assertGreater(np.linalg.norm(quadrature), 1e-3)

    def test_mc_uses_supplied_averages_without_evaluating_rhs(self):
        mesh = MeshTri().refined(1)
        u = np.zeros(mesh.p.shape[1])
        vols, hs, _, _ = element_geometry(mesh)
        fbar = np.linspace(-1, 1, mesh.t.shape[1])

        def unused_rhs(x):
            raise AssertionError("MC estimator must use its sampled averages")

        eta = residual_estimator(mesh, u, unused_rhs, fbar=fbar, quadrature_order=10)
        np.testing.assert_allclose(eta**2, hs**2 * vols * fbar**2)

    def test_constant_load_and_jump_terms_agree_across_methods(self):
        for mesh in (MeshTri().refined(1), MeshTet().refined(1)):
            with self.subTest(dim=mesh.p.shape[0]):
                u = np.random.default_rng(123).normal(size=mesh.p.shape[1])
                midpoint = residual_estimator(mesh, u, constant_one, quadrature_rule="midpoint")
                quadrature = residual_estimator(mesh, u, constant_one, quadrature_order=2)
                mc = residual_estimator(mesh, u, constant_one, fbar=np.ones(mesh.t.shape[1]))
                np.testing.assert_allclose(quadrature, midpoint)
                np.testing.assert_allclose(mc, midpoint)


if __name__ == "__main__":
    unittest.main()
