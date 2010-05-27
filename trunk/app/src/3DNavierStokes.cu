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

IncompressibleCustomSolver::IncompressibleCustomSolver() 
: Eqn_IncompressibleNS3DF(), m_simplexNoiseField(0), m_currentTime(0.f), m_simplexNoiseFieldSize(512)
{
  initializeSimplexNoise();
}

IncompressibleCustomSolver::~IncompressibleCustomSolver()
{}

void IncompressibleCustomSolver::initializeSimplexNoise()
{
  cudaMalloc((void**)&m_simplexNoiseField,sizeof(float)*m_simplexNoiseFieldSize*m_simplexNoiseFieldSize);
}


__global__ void addZForce(float *dwdt, float coefficient, float* noise, int2 noiseSize,
                          int xstride, int ystride, int nbr_stride, 
                          int nx, int ny, int nz, int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // shift so we will get maximum coalescing.  This means that we will need to test if k>0 below.
  int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;

  // use gaussian spreading out from middle
  if (i < nx && j < ny && k < nz) {
    float noiseValue = noise[idx];
    noiseValue *= 0.25f;
    noiseValue += 0.15f;

    float yRatio = -3.f*(1.f-j/ny)+2.f; // 2 at bottom, -1 at top
    yRatio *= 0.5f;
    float output = noiseValue*yRatio*coefficient;
    if (k > nx / 2)
      output *= -1.f;
    dwdt[idx] += output;
  }
}

__global__ void addXForce(float *dudt, float coefficient, float* noise, int2 noiseSize,
                          int xstride, int ystride, int nbr_stride, 
                          int nx, int ny, int nz, int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // shift so we will get maximum coalescing.  This means that we will need to test if k>0 below.
  int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;

  // use gaussian spreading out from middle
  if (i < nx && j < ny && k < nz) {
    float noiseValue = noise[idx];
    noiseValue *= 0.25f;
    noiseValue += 0.15f;

    float xRatio = 2.f*fabs(nx*0.5f-i)/nx; // 0 to 1 moving away from center
    xRatio *= 5.f;
    float yRatio = -3.f*(1.f-j/ny)+2.f; // 2 at bottom, -1 at top
    yRatio *= 0.5f;
    float output = noiseValue*yRatio*coefficient;
    if (i > nx / 2)
      output *= -1.f;
    dudt[idx] += output;
  }
}

__global__ void addVerticalForce(float *dvdt, float coefficient, float* noise, int2 noiseSize,
                                 int xstride, int ystride, int nbr_stride, 
                                 int nx, int ny, int nz, int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // shift so we will get maximum coalescing.  This means that we will need to test if k>0 below.
  int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;


  // use gaussian spreading out from middle
  if (i < nx && j < ny && k < nz) {
    float noiseValue = noise[idx];
    noiseValue *= 0.75f;
    noiseValue += 1.0f;

    float xRatio = powf(1-(2.f*fabs(nx*0.5f - i)/nx),2);
    float zRatio = powf(1-(2.f*fabs(nz*0.5f - k)/nz),2);
    float yRatio = (0.5f-(j/ny))*10.f;
    dvdt[idx] += noiseValue*yRatio*xRatio*zRatio*coefficient;//((float).5) * coefficient * (temperature[idx] + temperature[idx-nbr_stride]);
  }
}

void IncompressibleCustomSolver::add_external_forces(double dt)
{
  // apply thermal force by adding -gkT to dvdt (let g = -1, k = 1, so this is just dvdt += T)
  //_advection_solver.deriv_vdt.linear_combination((T)1.0, _advection_solver.deriv_vdt, (T)1.0, _thermal_solver.phi);

  int tnx = nz();
  int tny = ny();
  int tnz = nx();

  int threadsInX = 16;
  int threadsInY = 2;
  int threadsInZ = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  float *v = &_deriv_vdt.at(0,0,0);
  float *u = &_deriv_udt.at(0,0,0);
  float *w = &_deriv_wdt.at(0,0,0);

  int2 noiseSize = make_int2(m_simplexNoiseFieldSize,m_simplexNoiseFieldSize);

  float coefficient = 20.f;
  addVerticalForce<<<Dg, Db>>>(v, coefficient,m_simplexNoiseField, noiseSize,
    _temp.xstride(), _temp.ystride(), _temp.stride(DIR_YAXIS_FLAG), nx(), ny(), nz(), 
    blocksInY, 1.0f / (float)blocksInY);
  coefficient = 20.f;
  addXForce<<<Dg, Db>>>(u, coefficient,m_simplexNoiseField, noiseSize,
    _temp.xstride(), _temp.ystride(), _temp.stride(DIR_XAXIS_FLAG), nx(), ny(), nz(), 
    blocksInY, 1.0f / (float)blocksInY);
  coefficient = 20.f;
  addZForce<<<Dg, Db>>>(w, coefficient,m_simplexNoiseField, noiseSize,
    _temp.xstride(), _temp.ystride(), _temp.stride(DIR_ZAXIS_FLAG), nx(), ny(), nz(), 
    blocksInY, 1.0f / (float)blocksInY);

}
// custom update method
bool IncompressibleCustomSolver::advance_one_step(double dt)
{
  clear_error();
  num_steps++;
  m_currentTime += float(dt);

  // update dudt
  check_ok(_advection_solver.solve()); // updates dudt, dvdt, dwdt, overwrites whatever is there

  if (viscosity_coefficient() > 0) {
    check_ok(_u_diffusion.solve()); // dudt += \nu \nabla^2 u
    check_ok(_v_diffusion.solve()); // dvdt += \nu \nabla^2 v
    check_ok(_w_diffusion.solve()); // dwdt += \nu \nabla^2 w
  }

  //static float addingUserForcesSum = 0.f;
  //CPUTimer addingUserForcesTimer;
  //cudaThreadSynchronize();
  //addingUserForcesTimer.start();
  m_simplexNoise.updateNoise(m_simplexNoiseField, 0.0f, m_currentTime, 0.1f, m_simplexNoiseFieldSize*m_simplexNoiseFieldSize, m_simplexNoiseFieldSize);
  // eventually this will be replaced with a grid-wide operation.
  add_external_forces(dt);
  // advance u,v,w
  check_ok(_u.linear_combination((float)1.0, _u, (float)dt, _deriv_udt));
  check_ok(_v.linear_combination((float)1.0, _v, (float)dt, _deriv_vdt)); 
  check_ok(_w.linear_combination((float)1.0, _w, (float)dt, _deriv_wdt));

  //cudaThreadSynchronize();
  //addingUserForcesTimer.stop();
  //addingUserForcesSum += addingUserForcesTimer.elapsed_sec();
  //printf("user added forces: %f\n",addingUserForcesSum / (m_currentTime/dt));

  //static float projection3DSum = 0.f;
  //CPUTimer projection3DTimer;
  //cudaThreadSynchronize();
  //projection3DTimer.start();
  // enforce incompressibility - this enforces bc's before and after projection
  check_ok(_projection_solver.solve(_max_divergence));
  //cudaThreadSynchronize();
  //projection3DTimer.stop();
  //projection3DSum += projection3DTimer.elapsed_sec();
  //printf("average projection OpenCurrent: %f\n",projection3DSum / (m_currentTime/dt));

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
  m_currentTime = 0.f;
}

void NavierStokes3D::setupParams()
{
  params.init_grids(nx, ny, nz);
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
        float temp = fabs(0.2f*nz-k) / (0.2f*nz) * 1.f;
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
}

#include <iostream>
#include <fstream>

void NavierStokes3D::run(double dt)
{
  m_currentTime += float(dt);
  if (firstRun)
  {
    setupParams();
    firstRun = false;
  }

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
      hposx.at(p) += curVel.x * float(dt);
      hposy.at(p) += curVel.y * float(dt);
      hposz.at(p) += curVel.z * float(dt);
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
}

void NavierStokes3D::calculateVelocities(float dt)
{
}