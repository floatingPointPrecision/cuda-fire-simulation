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
@file SliceManager.cu
@note This file implements the SliceManager class
*/

#include "SliceManager.h"
#include "XMLParser.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace std;

SliceManager::SliceManager(const char* settingsFileName)
{
  // scalar value initialization
  m_numSliceSimulations = 1;
  XMLParser settingsFile(settingsFileName);
  settingsFile.getInt("numSlices",&m_numSliceSimulations);
  settingsFile.getInt("imageSize",&m_imageDim);
  m_domainSize = m_imageDim * m_imageDim;
  m_complexPadWidth = (m_imageDim/2+1);
  m_realPadWidth = (2*(m_imageDim/2+1));
  m_paddedDomainSize = (m_imageDim*m_complexPadWidth);
  settingsFile.getFloat("timestep",&m_dt);
  settingsFile.getFloat("viscosityCoefficient",&m_viscosityCoefficient);
  settingsFile.getFloat("massDissipationFactor",&m_massDissipationFactor);
  settingsFile.getFloat("fuelDissipationFactor",&m_fuelDissipationFactor);
  settingsFile.getFloat("coolingCoefficient",&m_coolingCoefficient);
  settingsFile.getFloat("maxTemperature",&m_maxTemperature);
  settingsFile.getFloat("maxTemperature",&m_combustionTemperature);
  settingsFile.getFloat("densityFactor",&m_densityFactor);
  settingsFile.getFloat("velocityRetention",&m_velocityRetention);
  settingsFile.getFloat("textureScale1",&m_textureScale.x);
  settingsFile.getFloat("textureScale2",&m_textureScale.y);
  settingsFile.getFloat("textureScale3",&m_textureScale.z);
  settingsFile.getFloat("textureScale4",&m_textureScale.w);
  settingsFile.getFloat("textureTimeScale",&m_textureTimeScale);

  float domainZSpace[2];
  settingsFile.setNewRoot("boundingBox");
  settingsFile.getFloat2("zRange",domainZSpace);
  m_frontZ = domainZSpace[0];
  m_backZ = domainZSpace[1];
  settingsFile.resetRoot();
  m_time = 0.f;
  m_pauseSimulation = true;

  // Simplex noise initialization
  m_simplexFieldTexture = new SimplexNoise4D;
  m_simplexFieldTurbulence = new SimplexNoise4D;
  // device utiltiy field initialization
  cudaMalloc((void**)&m_vxfield, sizeof(float2) * m_paddedDomainSize);
  cudaMalloc((void**)&m_vyfield, sizeof(float2) * m_paddedDomainSize);
  cudaMalloc((void**)&m_utilityScalarField, sizeof(float) * m_domainSize);
  cudaMalloc((void**)&m_turbulenceField, sizeof(float) * m_domainSize);
  cudaMalloc((void**)&m_phiField, sizeof(float) * m_domainSize);
  // host utiltiy field initialization
  m_utilityScalarField_h = (float*) malloc(sizeof(float) * m_domainSize);
  m_tempVelocityField_h = (float2*) malloc(sizeof(float2) * m_domainSize);
  m_frameBuffer_h = (int*) malloc(sizeof(int) * m_domainSize);
  // fft handles
  cufftPlan2d(&m_planr2c, m_imageDim, m_imageDim, CUFFT_R2C);
  cufftPlan2d(&m_planc2r, m_imageDim, m_imageDim, CUFFT_C2R);

  // slice initialization
  m_sliceSimulations = (SliceSimulation**) malloc(sizeof(SliceSimulation*)*m_numSliceSimulations);
  for (int i = 0; i < m_numSliceSimulations; i++)
  {
    m_sliceSimulations[i] = new SliceSimulation(this);
  }
}

SliceManager::~SliceManager()
{
  for (int i = 0; i < m_numSliceSimulations; i++)
    delete m_sliceSimulations[i];
  delete m_sliceSimulations;
  delete m_simplexFieldTexture;
  delete m_simplexFieldTurbulence;
  delete m_utilityScalarField_h;
  delete m_tempVelocityField_h;
  delete m_frameBuffer_h;
  cufftDestroy(m_planr2c);
  cufftDestroy(m_planc2r);
  cudaFree(m_vxfield);
  cudaFree(m_vyfield);
  cudaFree(m_utilityScalarField);
  cudaFree(m_turbulenceField);
  cudaFree(m_phiField);
}

void SliceManager::updateIndividualSlice(int sliceIndex, const float2* newVelocityField, const float* newMassField, const float* newFuelField)
{
  if (sliceIndex >= m_numSliceSimulations || sliceIndex < 0)
    return;
  float zIntercept = float (sliceIndex) * 0.25f; //m_backZ + ((m_frontZ-m_backZ)/m_numSliceSimulations)*sliceIndex;
  m_sliceSimulations[sliceIndex]->performSliceSimulation(newVelocityField, newMassField, newFuelField, zIntercept);
}

void SliceManager::displaySlice(int sliceIndex, int sliceType)
{
  if (sliceIndex >= m_numSliceSimulations || sliceIndex < 0)
    return;
  m_sliceSimulations[sliceIndex]->displaySlice(sliceType, m_pauseSimulation);
}

void SliceManager::writeDensityTemperatureToDisk(const char* fileName)
{
  FILE* outFile = fopen(fileName,"w");
  if (!outFile)
  {
    printf("unable to open file %s\n",fileName);
    return;
  }
  // write out the header which includes the image space dimensions
  int header[3] = {m_imageDim,m_imageDim,m_numSliceSimulations};
  int headerSize = sizeof(int)*3;
  fwrite(header,sizeof(int),3,outFile);  
  int densityStart = headerSize;
  for (int i = 0; i < m_numSliceSimulations; i++)
  {
    fseek(outFile,densityStart + i*sizeof(float)*m_domainSize,SEEK_SET);
    cudaMemcpy(m_utilityScalarField_h, m_sliceSimulations[i]->getDensityField(), sizeof(float)*m_domainSize, cudaMemcpyDeviceToHost);
    fwrite(m_utilityScalarField_h,sizeof(float),m_domainSize,outFile);
  }
  // write out temperature
  int temperatureStart = densityStart + sizeof(float)*m_domainSize*m_numSliceSimulations;
  for (int i = 0; i < m_numSliceSimulations; i++)
  {
    fseek(outFile,temperatureStart + i*sizeof(float)*m_domainSize,SEEK_SET);
    cudaMemcpy(m_utilityScalarField_h, m_sliceSimulations[i]->getTemperatureField(), sizeof(float)*m_domainSize, cudaMemcpyDeviceToHost);
    fwrite(m_utilityScalarField_h,sizeof(float),m_domainSize,outFile);
  }
  fclose(outFile);
}

void SliceManager::startUpdateSeries()
{
  m_time += m_dt;
}

float* SliceManager::getDensitySlab(int sliceIndex)
{
  if (sliceIndex >= m_numSliceSimulations || sliceIndex < 0)
    return 0;
  return m_sliceSimulations[sliceIndex]->getDensityField();
}

float* SliceManager::getTemperatureSlab(int sliceIndex)
{
  if (sliceIndex >= m_numSliceSimulations || sliceIndex < 0)
    return 0;
  return m_sliceSimulations[sliceIndex]->getTemperatureField();
}

float* SliceManager::getTextureSlab(int sliceIndex)
{
  if (sliceIndex >= m_numSliceSimulations || sliceIndex < 0)
    return 0;
  return m_sliceSimulations[sliceIndex]->getTextureField();
}

float* SliceManager::getFuelSlab(int sliceIndex)
{
  if (sliceIndex >= m_numSliceSimulations || sliceIndex < 0)
    return 0;
  return m_sliceSimulations[sliceIndex]->getFuelField();
}