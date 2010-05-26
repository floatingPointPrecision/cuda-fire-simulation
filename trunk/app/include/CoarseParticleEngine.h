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
@file CoarseParticleEngine.h
@brief Defines the CoarseParticleEngine class used for coarse particle simulation.
*/

#pragma once

// includes, system
#include <vector>
#include <GL/glew.h>
#include <GL/glut.h>

// includes, CUDA

// includes, CUFIRE
#include "CoarseParticleEngineDefinitions.h"
#include "3DNavierStokes.h"
//#include "Projection.h"

namespace cufire
{

  /**
  *  Coarse Particle Engine. Manager for the coarse particle simulation of broad fuel movement.
  */
  class CoarseParticleEngine
  {    

    // public methods
  public:
    /**
    * constructor. Sets the maximum number of particles and initializes necessary vectors on host and device
    * @param maxNumberParticles maximum number of particles in the system
    */
    CoarseParticleEngine(int maxNumberParticles, float2 xBBox, float2 yBBox, float2 zBBox);
    /**
    * A destructor.
    */
    ~CoarseParticleEngine();
    /**
    * advances simulation. Advances the particle simulation by the given timestep.
    * @param timestep time in seconds to advance the simulation
    */
    void advanceSimulation(float timestep);
    /**
    * renders particles. Renders the particles as OpenGL billboards.
    */
    void render();
    /**
    * adds particle. Adds particle to the host but does not add it to the device until flushParticles is called
    * @param position position in world space of the new particle
    * @param velocity velocty in world space of the new particle
    * @param fuel normalized fuel value for the new particle. Specifies how much contribution particle adds to fuel projections
    * @param radius radius of the particle
    * @param lifetime lifetime in seconds of the particle
    * @param mass normalized mass of the particle. Specifies how much contribution particle adds to mass projections
    * @param impulse normalized impulse value of the particle. Specifies how much contribution particle adds to impulse projections
    */
    void addParticle(float3 position, float3 velocity, float fuel, float radius, float lifetime, float mass, float impulse);
    /**
    * adds random particle. Adds random particle to the host but does not add it to the device until flushParticles is called
    * @param xBounds min and max x value for the particle's starting point
    * @param yBounds min and max y value for the particle's starting point
    * @param zBounds min and max z value for the particle's starting point
    */
    void addRandomParticle(float2 xBounds, float2 yBounds, float2 zBounds, int numParticles=1);
    /**
    * flushes particles to the GPU. Transfers all local particles to the GPU, used to group large transfers together and allow
    * incremental addition of particles
    */
    void flushParticles();
    /**
    * returns a struct filled with the begin iterators of all the particle attribute device vectors
    */
    ParticleItrStruct getParticleBegins() {return m_particleStructItrBegin;}
    /**
    * returns the number of particles
    */
    int getNumParticles() {return m_numParticles;}

    void enableCUDAVbo();
    void disableCUDAVbo();


    // protected methods
  protected:
    /**
    * initializes particles on host and GPU. Initializes the host and device vectors for all the individual particle components
    */
    void initializeParticles();
    /**
    * bins particle velocities. Bins particle velocities for non-divergence update.
    */
    void binParticles();
    /**
    * enforces non-divergent velocities. Enforces non-divergent velocity on the bins using OpenCurrent.
    */
    void enforceDivergence();
    /**
    * projects velocities to 2D slices. Projects particle velocities onto given slices with depth weight per slice.
    */
    void projectToSlices();

    // protected members
  protected:
    /**
    * resets particles. removes all particles and will eventually restart the system
    */
    void resetParticles();

    unsigned int m_maxNumParticles; ///< max number of particles in the system.
    unsigned int m_numParticles; ///< current number of particles in the system.
    float m_currentTime;
    float2 m_xBBox; ///< x range of simulation
    float2 m_yBBox; ///< y range of simulation
    float2 m_zBBox; ///< z range of simulation
    NavierStokes3D* m_nsSolver;
    bool m_firstTime;

    float4* m_hostPositionAge; ///< particle host copy of position x, y, z, age
    float4* m_hostFuelRadiusMassImpulse; ///< particle host copy of fuel, radius, mass, impulse
    float* m_hostXVelocities; ///< particle x velocities host copy
    float* m_hostYVelocities; ///< particle y velocities host copy
    float* m_hostZVelocities; ///< particle z velocities host copy

    GLuint m_positionsAgeVBO; ///< OpenGL vertex buffer object of the positionAge float4 array
    cudaGraphicsResource* m_positionsVBO_CUDA;
    float4* m_devicePositionAge; ///< particle device copy of position x, y, z, age
    float4* m_deviceFuelRadiusMassImpulse; ///< particle device copy of fuel, radius, mass, impulse
    float* m_deviceXVelocities; ///< particle x velocities device copy
    float* m_deviceYVelocities; ///< particle y velocities device copy
    float* m_deviceZVelocities; ///< particle z velocities device copy

    ParticleItrStruct m_particleStructItrBegin; ///< beginning of particle tuple of iterators

    /*thrust::host_vector<float4> m_hostPositionAge; ///< particle host copy of position x, y, z, age
    thrust::host_vector<float4> m_hostFuelRadiusMassImpulse; ///< particle host copy of fuel, radius, mass, impulse
    thrust::host_vector<float> m_hostXVelocities; ///< particle x velocities host copy
    thrust::host_vector<float> m_hostYVelocities; ///< particle y velocities host copy
    thrust::host_vector<float> m_hostZVelocities; ///< particle z velocities host copy

    GLuint m_positionsAgeVBO; ///< OpenGL vertex buffer object of the positionAge float4 array
    DevPtrFloat4 m_positionsAgeRaw; ///< temporary raw float4 pointer to OpenGL positionsAge VBO
    thrust::device_vector<float4> m_devicePositionAge; ///< particle device copy of position x, y, z, age
    thrust::device_vector<float4> m_deviceFuelRadiusMassImpulse; ///< particle device copy of fuel, radius, mass, impulse
    thrust::device_vector<float> m_deviceXVelocities; ///< particle x velocities device copy
    thrust::device_vector<float> m_deviceYVelocities; ///< particle y velocities device copy
    thrust::device_vector<float> m_deviceZVelocities; ///< particle z velocities device copy

    ParticleItrStruct m_particleStructItrBegin; ///< beginning of particle tuple of iterators
    ParticleItrStruct m_particleStructItrEnd; ///< end of particle tuple of iterators
    ParticleZipItr m_particleItrBegin; ///< beginning of particle zip_iterator
    ParticleZipItr m_particleItrEnd; ///< end of particle zip_iterator*/

    //NavierStokes3D m_ns3DSolver;
  };

}