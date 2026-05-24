from __future__ import annotations
from skfem import BilinearForm, LinearForm
from skfem.helpers import dot, grad

@BilinearForm
def laplace(u, v, w):
    return dot(grad(u), grad(v))

def quadrature_load_form(rhs):
    @LinearForm
    def load(v, w):
        return rhs(w.x) * v
    return load
