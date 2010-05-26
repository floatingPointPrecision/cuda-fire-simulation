/********************************************************************************
Copyright (c) 2010, Steve Lesser
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2)Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3) The name of contributors may not be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL STEVE LESSER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/**
@file 3DNavierStokes.cu
*/

#include "3DNavierStokes.h"
#include <cutil_math.h>

#include <algorithm>
#include "ocuequation/eqn_incompressns3d.h"  
#include "ocuutil/float_routines.h"
#include "ocustorage/grid3dboundary.h"



using namespace ocu;

IncompressibleCustomSolver::IncompressibleCustomSolver() : Eqn_IncompressibleNS3DF()
{}

IncompressibleCustomSolver::~IncompressibleCustomSolver()
{}

// not working right now
bool IncompressibleCustomSolver::init_derivs(int nx, int ny, int nz)
{
  Grid3DHost<float> deriv_udt_host,deriv_vdt_host,deriv_wdt_host;
  if (!deriv_udt_host.init_congruent(_deriv_udt,false) ||
      !deriv_vdt_host.init_congruent(_deriv_vdt,false) ||
      !deriv_wdt_host.init_congruent(_deriv_wdt,false))
  {
    printf("force grid failed\n");
    exit(-1);
  }
  int i,j,k;
  float weight = 0.1f;
  for (i=0; i < nx; i++)
  {
    float xRatio = fabs(0.5*nx-i) / (0.5*nx);
    for (j=0; j < ny; j++)
    {
      float yRatio = fabs(0.5*ny-j) / (0.5*ny);
      for (k=0; k < nz; k++) {
        float zRatio = fabs(0.5*nz-k) / (0.5*nz);
        deriv_udt_host.at(i,j,k) = xRatio * weight;
        deriv_vdt_host.at(i,j,k) = weight;//(j < ny / 2) ? 10 : -10;
        deriv_wdt_host.at(i,j,k) = zRatio * weight;
      }
    }
  }
  _deriv_udt.copy_all_data(deriv_udt_host);
  _deriv_vdt.copy_all_data(deriv_udt_host);
  _deriv_wdt.copy_all_data(deriv_udt_host);
  return true;
}

// custom update method
bool IncompressibleCustomSolver::advance_one_step(double dt)
{
  clear_error();
  num_steps++;

  // update dudt
  check_ok(_advection_solver.solve()); // updates dudt, dvdt, dwdt, overwrites whatever is there

  if (viscosity_coefficient() > 0) {
    check_ok(_u_diffusion.solve()); // dudt += \nu \nabla^2 u
    check_ok(_v_diffusion.solve()); // dvdt += \nu \nabla^2 v
    check_ok(_w_diffusion.solve()); // dwdt += \nu \nabla^2 w
  }

  // eventually this will be replaced with a grid-wide operation.
  add_thermal_force();

  // update dTdt

  check_ok(_thermal_solver.solve());   // updates dTdt, overwrites whatever is there
  if (thermal_diffusion_coefficient() > 0) {
    check_ok(_thermal_diffusion.solve()); // dTdt += k \nabla^2 T
  }

  float ab_coeff = -dt*dt / (2 * _lastdt);

  // advance T 
  if (_time_step == TS_ADAMS_BASHFORD2 && _lastdt > 0) {
    check_ok(_temp.linear_combination((float)1.0, _temp, (float)(dt - ab_coeff), _deriv_tempdt));
    check_ok(_temp.linear_combination((float)1.0, _temp, (float)ab_coeff, _last_deriv_tempdt));
  } 
  else {
    check_ok(_temp.linear_combination((float)1.0, _temp, (float)dt, _deriv_tempdt));
  }

  check_ok(apply_3d_boundary_conditions_level1_nocorners(_temp, _thermalbc, _hx, _hy, _hz));

  // advance u,v,w
  if (_time_step == TS_ADAMS_BASHFORD2 && _lastdt > 0) {
    check_ok(_u.linear_combination((float)1.0, _u, (float)(dt - ab_coeff), _deriv_udt));
    check_ok(_u.linear_combination((float)1.0, _u, (float)ab_coeff, _last_deriv_udt));

    check_ok(_v.linear_combination((float)1.0, _v, (float)(dt - ab_coeff), _deriv_vdt));
    check_ok(_v.linear_combination((float)1.0, _v, (float)ab_coeff, _last_deriv_vdt));

    check_ok(_w.linear_combination((float)1.0, _w, (float)(dt - ab_coeff), _deriv_wdt));
    check_ok(_w.linear_combination((float)1.0, _w, (float)ab_coeff, _last_deriv_wdt));

  }
  else {
    check_ok(_u.linear_combination((float)1.0, _u, (float)dt, _deriv_udt));
    check_ok(_v.linear_combination((float)1.0, _v, (float)dt, _deriv_vdt)); 
    check_ok(_w.linear_combination((float)1.0, _w, (float)dt, _deriv_wdt));
  }

  // copy state for AB2
  if (_time_step == TS_ADAMS_BASHFORD2) {
    _lastdt = dt;
    _last_deriv_tempdt.copy_all_data(_deriv_tempdt);
    _last_deriv_udt.copy_all_data(_deriv_udt);
    _last_deriv_vdt.copy_all_data(_deriv_vdt);
    _last_deriv_wdt.copy_all_data(_deriv_wdt);
  }

  // enforce incompressibility - this enforces bc's before and after projection
  check_ok(_projection_solver.solve(_max_divergence));

  return !any_error();

}



void NavierStokes3D::setGridDimensions(int x, int y, int z)
{
  nx = x;
  ny = y;
  nz = z;
}

void NavierStokes3D::setParticles(float4* positions, float* xVel, float* yVel, float* zVel, int numParticles)
{
  m_positions = positions;
  m_xVel = xVel;
  m_yVel = yVel;
  m_zVel = zVel;
  m_numParticles = numParticles;
}

void NavierStokes3D::allocate_particles(Grid1DHostF &hposx, Grid1DHostF &hposy, Grid1DHostF &hposz, Grid1DHostF &hvx, Grid1DHostF &hvy, Grid1DHostF &hvz,
                                        Grid1DDeviceF &posx, Grid1DDeviceF &posy, Grid1DDeviceF &posz, Grid1DDeviceF &vx, Grid1DDeviceF &vy, Grid1DDeviceF &vz, 
                                        float xsize, float ysize, float zsize)
{
  hposx.init(m_numParticles,0);
  hposy.init(m_numParticles,0);
  hposz.init(m_numParticles,0);
  hvx.init(m_numParticles,0);
  hvy.init(m_numParticles,0);
  hvz.init(m_numParticles,0);

  posx.init(m_numParticles,0);
  posy.init(m_numParticles,0);
  posz.init(m_numParticles,0);
  vx.init(m_numParticles,0);
  vy.init(m_numParticles,0);
  vz.init(m_numParticles,0);

  for (int p=0; p < m_numParticles; p++) {
    float4 currentPos = m_positions[p];
    hposx.at(p) = currentPos.x;
    hposy.at(p) = currentPos.y;
    hposz.at(p) = currentPos.z;
  }

  posx.copy_all_data(hposx);
  posy.copy_all_data(hposy);
  posz.copy_all_data(hposz);
}

NavierStokes3D::NavierStokes3D()
{
  firstRun = true;
}

void NavierStokes3D::setupParams()
{
  params.init_grids(nx, ny, nz);
  //eqn.init_derivs(nx,ny,nz);
  params.hx = 1;
  params.hy = 1;
  params.hz = 1;
  BoundaryCondition closed;
  closed.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  params.flow_bc = BoundaryConditionSet(closed);
  BoundaryCondition neumann;
  neumann.type = BC_NEUMANN;
  params.temp_bc = BoundaryConditionSet(neumann);
  int i,j,k;
  params.init_temp.clear_zero();

  for (i=0; i < nx; i++)
  {
    for (j=0; j < ny; j++)
    {
      for (k=0; k < nz; k++) {
        float temp = fabs(0.2*nz-k) / (0.2*nz) * 1.f;
        params.init_u.at(i,j,k) = 0;//(i < nx / 2) ? 5 : -5;
        params.init_v.at(i,j,k) = 0;//(j < ny / 2) ? 10 : -10;
        params.init_w.at(i,j,k) = 0;//(k < nx / 2) ? 5 : -5;
        params.init_temp.at(i,j,k) = (i < nx/2) ? -temp : temp;
      }
    }
  }
  params.max_divergence = 1e-3;
  if(!eqn.set_parameters(params))
  {
    printf("OpenCurrent parameters not properly set\n");
    exit(1);
  }
  allocate_particles(hposx, hposy, hposz, hvx, hvy, hvz, posx, posy, posz, vx, vy, vz, nx, ny, nz);
  //eqn.init_derivs(nx,ny,nz);
  
}

void NavierStokes3D::run(double dt)
{
  if (firstRun)
  {
    setupParams();
    firstRun = false;
  }

  CPUTimer timer;
  timer.start();

  for (int i = 0; i < 1; i++)
  {
    if(!eqn.advance(dt))
    {
      printf("OpenCurrent parameters not properly set\n");
      exit(1);
    }
    // trace points
    sample_points_mac_grid_3d(vx, vy, vz, posx, posy, posz, eqn.get_u(), eqn.get_v(), eqn.get_w(), params.flow_bc, 1,1,1);
    hvx.copy_all_data(vx); hvy.copy_all_data(vy); hvz.copy_all_data(vz);
    for (int p=0; p < hvx.nx(); p++) {
      float3 curVel = make_float3(hvx.at(p),hvy.at(p),hvz.at(p));
      // forward Euler
      hposx.at(p) += curVel.x * dt;
      hposy.at(p) += curVel.y * dt;
      hposz.at(p) += curVel.z * dt;
    }
    // copy positions back to device
    posx.copy_all_data(hposx); posy.copy_all_data(hposy); posz.copy_all_data(hposz);
  }

  for (int p=0; p < m_numParticles; p++) {
    float4 currentPos = m_positions[p];
    currentPos.x = hposx.at(p);
    currentPos.y = hposy.at(p);
    currentPos.z = hposz.at(p);
    m_positions[p] = currentPos;

    float3 currentVelocity = make_float3(hvx.at(p),hvy.at(p),hvz.at(p));
    m_xVel[p] = currentVelocity.x;
    m_yVel[p] = currentVelocity.y;
    m_zVel[p] = currentVelocity.z;
  }

  timer.stop();
  printf("Elapsed: %f, or %f fps\n", timer.elapsed_sec(), 100 / timer.elapsed_sec());
}

void NavierStokes3D::calculateVelocities(float dt)
{
}