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
@file SliceSimulation.h
@note This file defines the SliceSimulation class
*/

#pragma once

#include <vector_types.h>
#include "SimplexNoise.h"
#include <cufft.h>
#include "SliceManager.h"

///< specifies one of the five different slabs
enum Slice
{
  SliceTexture = 0,
  SliceDensity,
  SliceFuel,
  SliceTemperature,
  SliceVelocity
};

class SliceManager;

class SliceSimulation
{
public:
  SliceSimulation(SliceManager* sliceManager);
  ~SliceSimulation();

  float* getDensityField();
  float* getTemperatureField();
  float* getTextureField();
  float* getFuelField();

  void performSliceSimulation(const float2* newVelocityField, const float* newMassField, const float* newFuelField, float zIntercept);
  void displaySlice(int slice, bool pauseSimulation);

protected:
  void advectVelocity(float2 *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
  void diffuseProject(float2 *vx, float2 *vy, int dx, int dy, float dt, float visc);
  void updateVelocity(float2 *v, float *vx, float *vy, int dx, int pdx, int dy);

  void addVelocity(const float2* velocityField);
  void dissipateDensity(float dt);
  void dissipateFuel(float dt);
  void coolTemperature(float dt);
  void contributeSlices(const float* mass, const float* fuel);
  void semiLagrangianAdvection(const float2* velocityField, float* scalarField, float* m_utilityScalarField, float dt);
  void addTextureDetail(float time, float zVal);
  void addTurbulenceVorticityConfinement(float time, float zVal, float dt);
  void simulateFluids(float dt);
  void enforveVelocityIncompressibility(float dt);

  void setupTexture(int x, int y);
  void bindTexture(void);
  void unbindTexture(void);
  void updateTexture(float2 *data, size_t wib, size_t h, size_t pitch);
  void deleteTexture(void);

protected:
  SliceManager* m_sliceManager;
  // slabs
  float2 *dvfield;
  float* densityField;
  float* temperatureField;
  float* textureField;
  float* fuelField;

  float zIntercept;

  size_t tPitch;
};