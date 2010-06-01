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
@file SliceManager.h
@note This file defines the SliceManager class
*/

#pragma once

#include "SimplexNoise.h"
#include "SliceSimulation.h"

// forward declaration
class SliceSimulation;

/**
*  SliceManager class. Manages multiple SliceSimulations and stores necessary fields and values
* to be shared across slices.
*/
class SliceManager
{
public:
  /**
  * constructor. Loads all necessary values from the given xml settings file
  * @param settingsFileName name of the xml file to load settings from
  */
  SliceManager(const char* settingsFileName);
  /**
  * A destructor.
  */
  ~SliceManager();
  /**
  * advances simulation for a single slice by as single timestep
  * @param sliceIndex index of slice to advance simulation for
  * @param newVelocityField device velocity field result of the coarse particle projection
  * @param newMassField device mass field result of the coarse particle projection
  * @param newFuelField device fuel field result of the coarse particle projection
  */
  void updateIndividualSlice(int sliceIndex, const float2* newVelocityField, const float* newMassField, const float* newFuelField);
  /**
  * displays slab type of the slice at the given index
  * @param sliceIndex index of slice to advance simulation for
  * @param sliceType Slice enum defining which slab to display
  */
  void displaySlice(int sliceIndex, int sliceType);
  /**
  * starts a new timestep for all slices, should be called once for each global timestep
  */
  void startUpdateSeries();
  /**
  * sets the pause state of the simulation
  * @param pauseSimulation true to pause the simulation, false otherwise
  */
  void setPauseState(bool pauseSimulation) {m_pauseSimulation = pauseSimulation;}
  /**
  * gets a pointer to the start of the density slab for a given slice
  * @param sliceIndex index of slice to retrieve the density slab from
  */
  float* getDensitySlab(int sliceIndex);
  /**
  * gets a pointer to the start of the temperature slab for a given slice
  * @param sliceIndex index of slice to retrieve the temperature slab from
  */
  float* getTemperatureSlab(int sliceIndex);
  /**
  * gets a pointer to the start of the texture slab for a given slice
  * @param sliceIndex index of slice to retrieve the texture slab from
  */
  float* getTextureSlab(int sliceIndex);
  /**
  * gets a pointer to the start of the fuel slab for a given slice
  * @param sliceIndex index of slice to retrieve the fuel slab from
  */
  float* getFuelSlab(int sliceIndex);
  /**
  * @return number of slice simulations
  */
  int getNumSlices()
  {
    return m_numSliceSimulations;
  }
  /**
  * @return pointer to the SimplexNoise4D used for texture slabs
  */
  SimplexNoise4D* getSimplexTexture() { return m_simplexFieldTexture;}
  /**
  * @return pointer to the SimplexNoise4D used for turbulence during vorticity confinement
  */
  SimplexNoise4D* getSimplexTurbulence() { return m_simplexFieldTurbulence;}
  /**
  * @return width of the (square) slabs
  */
  int getImageDim() {return m_imageDim;}
  /**
  * @return number of pixels in the slabs
  */
  int getDomainSize() {return m_domainSize;}
  /**
  * @return padded width for complex values used in the fft
  */
  int getComplexPadWidth() {return m_complexPadWidth;}
  /**
  * @return padded width for real values used in the fft
  */
  int getRealPadWidth() {return m_realPadWidth;}
  /**
  * @return padded width of a row in the image
  */
  int getPaddedDomainWidth() {return m_paddedDomainSize;}
  /**
  * @return how far the simulation advances each simulation step
  */
  float getTimestep() {return m_dt;}
  /**
  * @return the current total time in the simulation
  */
  float getTime() {return m_time;}
  /**
  * @return viscosity coefficient used to control damping velocity
  */
  float getViscosityCoefficient() {return m_viscosityCoefficient;}
  /**
  * @return mass dissipation amount M* = M * (1-mass dissipation)^dt
  */
  float getMassDissipationCoefficient() {return m_massDissipationFactor;}
  /**
  * @return fuel dissipation amount F* = F * (1-fuel dissipation)^dt
  */
  float getFuelDissipationCoefficient() {return m_fuelDissipationFactor;}
  /**
  * @return cooling coefficient
  */
  float getCoolingCoefficient() {return m_coolingCoefficient;}
  /**
  * @return maximum temperature of the system
  */
  float getMaxTemperature() {return m_maxTemperature;}
  /**
  * @return density factor
  */
  float getDensityFactor() {return m_densityFactor;}
  /**
  * @return amount of previous simulation steps velocity to retain
  */
  float getVelocityRetention() {return m_velocityRetention;}
  /**
  * @return the scales of the four textures, values of 0 are ignored
  */
  float4 getTextureScale() {return m_textureScale;}
  /**
  * @return amount to multiply texture time to control animation
  */
  float getTextureTimeScale() {return m_textureTimeScale;}
  // device field gets
  /**
  * @return complex velocity field in the x direction
  */
  float2* getVXField_d() {return m_vxfield;}
  /**
  * @return complex velocity field in the y direction
  */
  float2* getVYField_d() {return m_vyfield;}
  /**
  * @return general purpose device field the same size as slab fields
  */
  float* getUtilityScalarField_d() {return m_utilityScalarField;}
  /**
  * @return turbulence field used in vorticity confinement
  */
  float* getTurbulenceField_d() {return m_turbulenceField;}
  /**
  * @return phi field used in vorticity confinement
  */
  float* getPhiField_d() {return m_phiField;}
  /**
  * @return utility scalar field on the host
  */
  float* getUtilityScalarField_h() {return m_utilityScalarField_h;}
  /**
  * @return utility float2 field on the host
  */
  float2* getUtilityFloat2Field_h() { return m_tempVelocityField_h;}
  /**
  * @return int field on the host used as a framebuffer
  */
  int* getUtilityIntField_h() { return m_frameBuffer_h;}
  /**
  * @return handle to real to complex fft operation
  */
  // fft gets
  cufftHandle getRealToComplexFFT() {return m_planr2c;}
  /**
  * @return handle to complex to real fft operation
  */
  cufftHandle getComplexToRealFFT() {return m_planc2r;}

protected:
  // slice scalar attributes
  int m_numSliceSimulations; ///< number of slices in the simulation
  int m_imageDim; ///< width of a slice
  int m_domainSize; ///< number of values in a slice
  int m_complexPadWidth; ///< complex pad width used in fft
  int m_realPadWidth; ///< real pad width used in fft
  int m_paddedDomainSize; ///< padded domain size
  float m_dt; ///< length of a timestep
  float m_time; ///< current time in the simulation
  float m_viscosityCoefficient; ///< viscosity coefficient
  float m_massDissipationFactor; ///< mass dissipation factor
  float m_fuelDissipationFactor; ///< fuel dissipation factor
  float m_coolingCoefficient; ///< cooling coefficient
  float m_maxTemperature; ///< max system temperature
  float m_densityFactor; ///< density factor
  float m_combustionTemperature; ///< temperature of combustion
  float m_velocityRetention; ///< amount of velocity to retain from previous time slice
  float m_frontZ; ///< z value of the front of the system bounding box
  float m_backZ; ///< z value of the back of the system bounding box
  bool m_pauseSimulation; ///< whether to pause the simulation or not

  float4 m_textureScale; ///< scales of the four possible texture octaves
  float m_textureTimeScale; ///< amount to scale texture time to control animation

  // slice device utility fields
  float2* m_vxfield; ///< complex velocity field in x (device)
  float2* m_vyfield; ///< complex velocity field in y (device)
  float* m_utilityScalarField; ///< utility scalar field (device)
  float* m_turbulenceField; ///< turbulence scalar field (device)
  float* m_phiField; ///< phi scalar field (device)

  // slice host utility fields
  float* m_utilityScalarField_h; ///< utility float scalar field (host)
  float2* m_tempVelocityField_h; ///< utility float2 scalar field (host)
  int* m_frameBuffer_h; ///< Utility int scalar field (host)

  // fft handles
  cufftHandle m_planr2c; ///< handle to real to complex fft operation
  cufftHandle m_planc2r; ///< handle to complex to real fft operation

  // manager slices directly instead of vectors
  SliceSimulation** m_sliceSimulations; ///< array of SliceSimulation objects

  // device SimplexNoise fields
  SimplexNoise4D* m_simplexFieldTexture; ///< SimplexNoise4D used for texture slabs
  SimplexNoise4D* m_simplexFieldTurbulence; ///< SimplexNoise4D used for turbulence

};