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
@file Projection.h
*/

#pragma once

// includes, CUDA
// includes, CUDA
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/device_ptr.h"
#include "thrust/iterator/zip_iterator.h"

// includes, cufire
// (forward declarations)
#include "CoarseParticleEngineDefinitions.h"
#include "Command.h"

using namespace cufire;

namespace cufire
{
  /**
  *  Projection class, used for taking a 3D grid of particles and projecting them onto a view-oriented 2D area
  */
  class Projection
    : public Command
  {    
    // public methods
  public:
    /**
    * constructor. Initializes the Projection object
    */
    Projection(float3 gridCenter, float3 gridDimensions, float projectionDepth, int2 slicePixelDimensions);
    /**
    * A destructor.
    */
    virtual ~Projection(){};
    /**
    * executes the command
    */
    virtual void execute()=0;
    /**
    * Set the starting iterator for the particles and the number of particles
    */
    void setParticles(ParticleItrStruct particlesBegin, int numParticles) {m_particlesBegin = particlesBegin;m_numParticles = numParticles;} 

    // protected members
  protected:
    float3 m_gridCenter;
    float3 m_gridDims;
    float m_projectionDepth;
    int2 m_slicePixelDims;
    float4* m_outputSlice;
    int m_numParticles;

    ParticleItrStruct m_particlesBegin;
  };

  /**
  *  OrthographicProjection class, used for taking a 3D grid of particles and projecting them orthographically onto a view-oriented 2D area.
  *  Assumes the frustum front and back are perpendicular to the z-axis (so don't move the camera).
  */
  class OrthographicProjection
    : public Projection
  {
    // public methods
  public:
    /**
    * constructor. Initializes the OrthographicProjection class which is a Projection
    */
    OrthographicProjection(float3 gridCenter, float3 gridDimensions, float projectionDepth, int2 slicePixelDimensions, float2 sliceWorldDimensions);
    /**
    * destructor, does nothing
    */
    ~OrthographicProjection(){}
    /**
    * executes the command
    */
    void execute();
    /**
    * Sets the z-intercept for the next slice to project
    */
    void setSliceInformation(float zIntercept, float4* outputSlice);

    // protected members
  protected:
    float m_ZIntercept;
    float2 m_sliceWorldDims; 
  };
}