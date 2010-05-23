/*******************************************************************************
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
@file SolveNonDivergence.h
*/

#pragma once

#include "ocuequation/eqn_incompressns3d.h"
#include "ocustorage/grid1d.h"
#include "ocustorage/grid3dboundary.h"
#include "ocustorage/grid3dsample.h"
#include "ocuutil/timer.h"

using namespace ocu;

class NavierStokes3D
{
public:
  NavierStokes3D();

  void allocate_particles(Grid1DHostF &hposx, Grid1DHostF &hposy, Grid1DHostF &hposz, Grid1DHostF &hvx, Grid1DHostF &hvy, Grid1DHostF &hvz,
    Grid1DDeviceF &posx, Grid1DDeviceF &posy, Grid1DDeviceF &posz, Grid1DDeviceF &vx, Grid1DDeviceF &vy, Grid1DDeviceF &vz, 
    float xsize, float ysize, float zsize);
  void setGridDimensions(int x, int y, int z);
  void setParticles(float4* positions, float* xVel, float* yVel, float* zVel, int numParticles);

  void run();

private:
  void calculateVelocities(float dt);
  void setupParams();

  float4* m_positions;
  float* m_xVel;
  float* m_yVel;
  float* m_zVel;
  int m_numParticles;

  int nx;
  int ny;
  int nz;
  bool firstRun;

  Eqn_IncompressibleNS3DParamsF params;
  Eqn_IncompressibleNS3DF eqn;
  Grid1DHostF hposx, hposy, hposz;
  Grid1DHostF hvx, hvy, hvz;
  Grid1DDeviceF posx, posy, posz;
  Grid1DDeviceF vx, vy, vz;
};
