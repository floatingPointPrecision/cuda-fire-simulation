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
@file SliceSimulation.cu
@note This file implements the non-kernel methods of SliceSimulation
*/

/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation. 
* Any use, reproduction, disclosure, or distribution of this software 
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA) 
* associated with this source code for terms and conditions that govern 
* your use of this NVIDIA software.
* 
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <GL/glew.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <cutil_math.h>
#include "SliceSimulation.h"
#include "SliceSimulation_Kernels.cu"

#include <GL/glut.h>
#include "ocuutil/timer.h"

// CUDA example code that implements the frequency space version of 
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the 
// CUDA FFT library (CUFFT) to perform velocity diffusion and to 
// force non-divergence in the velocity field at each time step. It uses 
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step. 

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

float* SliceSimulation::getDensityField()
{
  return densityField;
}
float* SliceSimulation::getTemperatureField()
{
  return temperatureField;
}
float* SliceSimulation::getTextureField()
{
  return textureField;
}
float* SliceSimulation::getFuelField()
{
  return fuelField;
}

void SliceSimulation::performSliceSimulation(const float2* newVelocityField, const float* newMassField, const float* newFuelField, float zIntercept)
{
  float dt = m_sliceManager->getTimestep();
  float currentTime = m_sliceManager->getTime();
  bindTexture();
  addVelocity(newVelocityField);
  dissipateDensity(dt);
  dissipateFuel(dt);
  coolTemperature(dt);
  contributeSlices(newMassField, newFuelField);
  simulateFluids(dt);
  addTextureDetail(currentTime, zIntercept);
  addTurbulenceVorticityConfinement(currentTime, zIntercept, dt);
  enforveVelocityIncompressibility(dt);
  unbindTexture();
}

void SliceSimulation::advectVelocity(float2 *v, float *vx, float *vy,
                                     int dx, int pdx, int dy, float dt) 
{ 
  dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));
  dim3 tids(TIDSX, TIDSY);
  updateTexture(v, m_sliceManager->getImageDim()*sizeof(float2), m_sliceManager->getImageDim(), tPitch);
  advectVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, dt, TILEY/TIDSY);
  cutilCheckMsg("advectVelocity_k failed.");
}

void SliceSimulation::diffuseProject(float2 *vx, float2 *vy, int dx, int dy, float dt,
                                     float visc) 
{ 
  // Forward FFT
  cufftExecR2C(m_sliceManager->getRealToComplexFFT(), (cufftReal*)vx, (cufftComplex*)vx); 
  cufftExecR2C(m_sliceManager->getRealToComplexFFT(), (cufftReal*)vy, (cufftComplex*)vy);
  uint3 grid = make_uint3((dx/TILEX)+(!(dx%TILEX)?0:1), 
    (dy/TILEY)+(!(dy%TILEY)?0:1), 1);
  uint3 tids = make_uint3(TIDSX, TIDSY, 1);
  diffuseProject_k<<<grid, tids>>>(vx, vy, dx, dy, dt, visc, TILEY/TIDSY);
  cutilCheckMsg("diffuseProject_k failed.");
  // Inverse FFT
  cufftExecC2R(m_sliceManager->getComplexToRealFFT(), (cufftComplex*)vx, (cufftReal*)vx); 
  cufftExecC2R(m_sliceManager->getComplexToRealFFT(), (cufftComplex*)vy, (cufftReal*)vy);
}

void SliceSimulation::updateVelocity(float2 *v, float *vx, float *vy, 
                                     int dx, int pdx, int dy) 
{ 
  dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));
  dim3 tids(TIDSX, TIDSY);
  updateVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, TILEY/TIDSY,  tPitch);
  cutilCheckMsg("updateVelocity_k failed.");
}

void SliceSimulation::addVelocity(const float2* velocityField)
{
  int blockSize = 512;
  int gridSize = (m_sliceManager->getDomainSize()+ blockSize - 1) / blockSize;
  float amountVelocityToRetain = m_sliceManager->getVelocityRetention();
  addVelocityContributionKernel<<<gridSize,blockSize>>>(dvfield, velocityField, m_sliceManager->getImageDim(), 
    m_sliceManager->getDomainSize(), amountVelocityToRetain);
  // clamp velocities
  clampVelocities<<<gridSize,blockSize>>>(dvfield, 25.f, m_sliceManager->getDomainSize());
}

void SliceSimulation::dissipateDensity(float dt)
{
  int blockSize = 512;
  int gridSize = (m_sliceManager->getDomainSize()+ blockSize - 1) / blockSize;
  dissipateDenistyKernel<<<gridSize,blockSize>>>(densityField,m_sliceManager->getMassDissipationCoefficient(),dt,m_sliceManager->getDomainSize());
  cutilCheckMsg("dissipate density failed");
}

void SliceSimulation::dissipateFuel(float dt)
{
  int blockSize = 512;
  int gridSize = (m_sliceManager->getDomainSize()+ blockSize - 1) / blockSize;
  dissipateFuelKernel<<<gridSize,blockSize>>>(fuelField,m_sliceManager->getFuelDissipationCoefficient(),dt,m_sliceManager->getDomainSize());
  cutilCheckMsg("dissipate fuel failed");
}

void SliceSimulation::coolTemperature(float dt)
{
  int blockSize = 512;
  int gridSize = (m_sliceManager->getDomainSize()+ blockSize - 1) / blockSize;
  coolTemperatureKernel<<<gridSize,blockSize>>>(temperatureField,m_sliceManager->getCoolingCoefficient(),
    m_sliceManager->getMaxTemperature(),dt,m_sliceManager->getDomainSize());
  cutilCheckMsg("cool temperature failed");
}

void SliceSimulation::contributeSlices(const float* mass, const float* fuel)
{
  int blockSize = 512;
  int gridSize = (m_sliceManager->getDomainSize()+ blockSize - 1) / blockSize;
  addMassFromSlice<<<gridSize,blockSize>>>(densityField, mass, m_sliceManager->getDensityFactor(), m_sliceManager->getDomainSize());
  cutilCheckMsg("adding density from slice mass failed");
  addFuelFromSlice<<<gridSize,blockSize>>>(fuelField,fuel, m_sliceManager->getMaxTemperature(), m_sliceManager->getDomainSize());
  cutilCheckMsg("adding temperature from fuel slice failed");
  addTemperatureFromFuel<<<gridSize,blockSize>>>(temperatureField, fuelField, m_sliceManager->getMaxTemperature(), m_sliceManager->getDomainSize());
}

void SliceSimulation::semiLagrangianAdvection(const float2* velocityField, float* scalarField, float* m_utilityScalarField, float dt)
{
  int blockSize = 512;
  int gridSize = (m_sliceManager->getDomainSize()+ blockSize - 1) / blockSize;
  semiLagrangianAdvectionKernel<<<gridSize,blockSize>>>(velocityField, scalarField, m_utilityScalarField, dt, m_sliceManager->getDomainSize(), 
    m_sliceManager->getImageDim());
  cudaMemcpy(scalarField,m_utilityScalarField,sizeof(float)*m_sliceManager->getDomainSize(),cudaMemcpyDeviceToDevice);
  cutilCheckMsg("semi-Lagrangian advection");
}

void SliceSimulation::addTextureDetail(float time, float zVal)
{
  time *= m_sliceManager->getTextureTimeScale();
  float4 texScale = m_sliceManager->getTextureScale();
  if (texScale.x != 0.f)
    m_sliceManager->getSimplexTexture()->updateNoise(textureField, zVal, time, texScale.x, m_sliceManager->getDomainSize(), m_sliceManager->getImageDim());
  if (texScale.y != 0.f)
    m_sliceManager->getSimplexTexture()->updateNoise(textureField, zVal, time, texScale.y, m_sliceManager->getDomainSize(), m_sliceManager->getImageDim());
  if (texScale.z != 0.f)
    m_sliceManager->getSimplexTexture()->updateNoise(textureField, zVal, time, texScale.z, m_sliceManager->getDomainSize(), m_sliceManager->getImageDim());
  if (texScale.w != 0.f)
    m_sliceManager->getSimplexTexture()->updateNoise(textureField, zVal, time, texScale.w, m_sliceManager->getDomainSize(), m_sliceManager->getImageDim());
  cutilCheckMsg("texture synthesis");
}

void SliceSimulation::addTurbulenceVorticityConfinement(float time, float zVal, float dt)
{
  float* d_turbulenceField = m_sliceManager->getTurbulenceField_d();
  float* d_phiField = m_sliceManager->getPhiField_d();
  m_sliceManager->getSimplexTurbulence()->updateNoise(d_turbulenceField, time, zVal, 1.f, m_sliceManager->getDomainSize(), m_sliceManager->getImageDim());
  int blockSize = 512;
  int gridSize = (m_sliceManager->getDomainSize()+ blockSize - 1) / blockSize;
  float vorticityTerm = 150.f;
  calculatePhi<<<gridSize,blockSize>>>(dvfield,d_phiField, d_turbulenceField, vorticityTerm,m_sliceManager->getDomainSize(), m_sliceManager->getImageDim());
  vorticityConfinementTurbulence<<<gridSize,blockSize>>>(dvfield, d_phiField, dt,m_sliceManager->getDomainSize(), m_sliceManager->getImageDim());
}

void SliceSimulation::simulateFluids(float dt)//float2* newVelocityField)
{
  // perform advection
  advectVelocity(dvfield, (float*)m_sliceManager->getVXField_d(), (float*)m_sliceManager->getVYField_d(), m_sliceManager->getImageDim(), 
    m_sliceManager->getRealPadWidth(), m_sliceManager->getImageDim(), dt);
  dt *= 10.f;
  semiLagrangianAdvection(dvfield, densityField, m_sliceManager->getUtilityScalarField_d(), dt);
  semiLagrangianAdvection(dvfield, fuelField, m_sliceManager->getUtilityScalarField_d(), dt);
  semiLagrangianAdvection(dvfield, temperatureField, m_sliceManager->getUtilityScalarField_d(), dt*10.f);
  semiLagrangianAdvection(dvfield, textureField, m_sliceManager->getUtilityScalarField_d(), dt);
}

void SliceSimulation::enforveVelocityIncompressibility(float dt)
{
  diffuseProject(m_sliceManager->getVXField_d(), m_sliceManager->getVYField_d(), m_sliceManager->getComplexPadWidth(), m_sliceManager->getImageDim(),
    dt, m_sliceManager->getViscosityCoefficient());
  updateVelocity(dvfield, (float*)m_sliceManager->getVXField_d(), (float*)m_sliceManager->getVYField_d(), m_sliceManager->getImageDim(), 
    m_sliceManager->getRealPadWidth(), m_sliceManager->getImageDim());
}

void SliceSimulation::displaySlice(int slice, bool pauseSimulation)
{
  float pixelScale, pixelAddition, *field;
  float* h_scalarField = m_sliceManager->getUtilityScalarField_h();
  if (slice == SliceVelocity)
  {
    float2* h_velField = m_sliceManager->getUtilityFloat2Field_h();
    cudaMemcpy(h_velField,dvfield,sizeof(float2)*m_sliceManager->getDomainSize(),cudaMemcpyDeviceToHost);
    float output;
    float2 velocity;
    float xTotal = 0.f, yTotal = 0.f;
    int numNonEmpty = 0;
    for (int i = 0; i < m_sliceManager->getImageDim(); i++)
    {
      for (int j = 0; j < m_sliceManager->getImageDim(); j++)
      {
        velocity = h_velField[j*m_sliceManager->getImageDim()+i];
        velocity.x = fabs(velocity.x);
        velocity.y = fabs(velocity.y);
        if (velocity.x != 0.f || velocity.y != 0.f)
        {
          numNonEmpty++;
          if (velocity.x != 0.f)
            xTotal += velocity.x;
          if (velocity.y != 0.f)
            yTotal += velocity.y;
        }
        output = sqrtf(velocity.x*velocity.x+velocity.y*velocity.y);
        output /= 30.f;
        h_scalarField[j*m_sliceManager->getImageDim()+i] = output;
      }
    }
    if (!pauseSimulation)
    {
      printf("x average: %f\n", xTotal / numNonEmpty);
      printf("y average: %f\n", yTotal / numNonEmpty);
    }
    glWindowPos2i(0,0);
    glDrawPixels(m_sliceManager->getImageDim(),m_sliceManager->getImageDim(),GL_LUMINANCE,GL_FLOAT,h_scalarField);
    return;
  }
  else
  {
    if (slice == SliceTexture)
    {
      pixelAddition = 0.5f;
      pixelScale = 0.5f;
      field = textureField;
    }
    else if (slice == SliceFuel)
    {
      pixelAddition = 0.f;
      pixelScale = 1.f / m_sliceManager->getMaxTemperature();
      field = fuelField;
    }
    else if (slice == SliceDensity)
    {
      pixelAddition = 0.f;
      pixelScale = 1.f / m_sliceManager->getDensityFactor();
      field = densityField;
    }
    else if (slice == SliceTemperature)
    {
      pixelAddition = 0.f;
      pixelScale = 1.f / m_sliceManager->getMaxTemperature();
      field = temperatureField;
    }
    cudaMemcpy(h_scalarField,field,sizeof(float)*m_sliceManager->getDomainSize(),cudaMemcpyDeviceToHost);
    float total = 0.f;
    int numNonEmpty = 0;
    for (int i = 0; i < m_sliceManager->getImageDim(); i++)
    {
      for (int j = 0; j < m_sliceManager->getImageDim(); j++)
      {
        float output = h_scalarField[j*m_sliceManager->getImageDim()+i];
        if (output != 0.f)
        {
          numNonEmpty++;
          total += output;
        }
        output = output*pixelScale+pixelAddition;
        h_scalarField[j*m_sliceManager->getImageDim()+i] = output;
      }
    }
    if (!pauseSimulation)
      printf("average scalar field value: %f\n",total / numNonEmpty);
  }
  glWindowPos2i(0,0);
  glDrawPixels(m_sliceManager->getImageDim(),m_sliceManager->getImageDim(),GL_LUMINANCE,GL_FLOAT,h_scalarField);

}

SliceSimulation::~SliceSimulation() {
  // Free all host and device resources
  cudaFree(dvfield);
  cudaFree(densityField);
  cudaFree(temperatureField);
  cudaFree(textureField);
  cudaFree(fuelField);
}

SliceSimulation::SliceSimulation(SliceManager* sliceManager)
{
  m_sliceManager = sliceManager;
  // Allocate and initialize device data
  tPitch = 0;
  cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(float2)*m_sliceManager->getImageDim(), m_sliceManager->getImageDim());
  cudaMemset(dvfield, 0, sizeof(float2) * m_sliceManager->getDomainSize());
  // Scalar slab fields
  cudaMalloc((void**)&densityField, sizeof(float) * m_sliceManager->getDomainSize());
  cudaMalloc((void**)&temperatureField, sizeof(float) * m_sliceManager->getDomainSize());
  cudaMalloc((void**)&textureField, sizeof(float) * m_sliceManager->getDomainSize());
  cudaMalloc((void**)&fuelField, sizeof(float) * m_sliceManager->getDomainSize());
  cudaMemset(densityField,0,sizeof(float)*m_sliceManager->getDomainSize());
  cudaMemset(temperatureField,0,sizeof(float)*m_sliceManager->getDomainSize());
  cudaMemset(textureField,0,sizeof(float)*m_sliceManager->getDomainSize());
  cudaMemset(fuelField,0,sizeof(float)*m_sliceManager->getDomainSize());

  setupTexture(m_sliceManager->getImageDim(), m_sliceManager->getImageDim());
}
