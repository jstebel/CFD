from fenics import *

# Solve advection equation by finite volume method:
#
#    div(v*phi) = f
#
# in unit square, with the boundary condition
#
#    phi(x, y) = 1             where v.n < 0.
#
# Return the computed solution.
def advection_fvm_steady(
      nx,         # number of elements in 1 direction
      v,          # velocity field
      f,          # volume source
      g,          # inlet boundary condition
      fname       # output file name
    ):

  # Create mesh
  mesh = UnitSquareMesh.create(nx, nx, CellType.Type.quadrilateral)

  # Define variational problem
  # The variational formulation reads:
  #
  #   Find phi in V such that for all w in V the following holds:
  #     sum_over_faces( integral_over_face((beta*average(phi) + |beta|/2*jump(phi))*jump(w))*dS )
  #     + integral_over_boundary((beta^+)*phi*w)*ds
  #     = integral_over_domain(f*w)*dx
  #     - integral_over_boundary((beta^-)*g*w)*ds,
  #
  # where beta = v.n, beta^+ = positive part of beta, beta^- = negative part of beta.
  V = FunctionSpace(mesh, "DG", 0)
  phi = TrialFunction(V)
  w = TestFunction(V)
  n = FacetNormal(mesh)
  beta = dot(v,n)("+")
  beta_plus = (dot(v,n)+abs(dot(v,n)))*0.5
  beta_minus= (dot(v,n)-abs(dot(v,n)))*0.5
  a = (beta*avg(phi) + abs(beta)*0.5*jump(phi))*jump(w)*dS + beta_plus*phi*w*ds
  l = f*w*dx - beta_minus*g*w*ds

  # Compute solution
  solution = Function(V)
  solution.rename("solution", "phi")
  solve(a == l, solution)
  f = File(fname)
  f << solution

  return solution




# Solve unsteady advection equation by finite volume method:
#
#    d phi/ dt + div(v*phi) = f
#
# in unit square, with the initial condition
#
#    phi(0, x, y) = init(x, y)
#
# and the boundary condition
#
#    phi(t, x, y) = 1             where v.n < 0.
#
# Return the computed solution.
def advection_fvm_unsteady(
      T,          # end time
      nsteps,     # number of time steps
      nx,         # number of elements in 1 direction
      v,          # velocity field
      f,          # volume source
      g,          # inlet boundary condition
      init,       # initial condition
      fname       # output file name
    ):

  # Create mesh
  mesh = UnitSquareMesh.create(nx, nx, CellType.Type.quadrilateral)

  # Define variational problem
  V = FunctionSpace(mesh, "DG", 0)
  phi = TrialFunction(V)
  phi0 = project(init, V)    # set initial condition
  w = TestFunction(V)
  n = FacetNormal(mesh)
  beta = dot(v,n)("+")
  beta_plus = (dot(v,n)+abs(dot(v,n)))*0.5
  beta_minus= (dot(v,n)-abs(dot(v,n)))*0.5
  dt = T/nsteps
  cdt = Constant(dt)
  a = phi*w*dx
  l = phi0*w*dx + dt*f*w*dx - dt*beta_minus*g*w*ds - (dt*(beta*avg(phi0) + abs(beta)*0.5*jump(phi0))*jump(w)*dS + dt*beta_plus*phi0*w*ds)

  # Compute solution
  solution = Function(V)
  solution.rename("solution", "phi")
  solution.assign(phi0)
  f = File(fname)
  f << (solution, 0)
  for nt in range(0,nsteps+1):
    solve(a == l, solution)
    f << (solution, nt*dt)
    phi0.vector()[:] = solution.vector()

  return phi0