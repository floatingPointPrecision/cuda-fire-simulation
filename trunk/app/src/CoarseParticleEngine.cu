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
@file CoarseParticleEngine.cu
*/

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>

// includes, CUDA
#include <cutil_inline.h>
#include <cutil_math.h>
#include <cuda_gl_interop.h>
#include <cutil_gl_inline.h>

// includes, CUFIRE
#include "CoarseParticleEngine.h"
#include "RandomUtilities.h"
#include "Projection.h"

using namespace cufire;

////////////////// HOST CODE ////////////////////
CoarseParticleEngine::CoarseParticleEngine(int maxNumberParticles, float2 xBBox, float2 yBBox, float2 zBBox)
: m_maxNumParticles(maxNumberParticles), m_currentTime(0.f), m_xBBox(xBBox), m_yBBox(yBBox), m_zBBox(zBBox)
{
  initializeParticles();
  m_firstTime = true;
  resetParticles();
}

CoarseParticleEngine::~CoarseParticleEngine()
{
  delete m_nsSolver;
  cudaGLUnregisterBufferObject(m_positionsAgeVBO);
  glDeleteBuffers(1, &m_positionsAgeVBO);
}

void CoarseParticleEngine::advanceSimulation(float timestep)
{
  m_currentTime += timestep;
  if (m_firstTime)
  {
    m_nsSolver = new NavierStokes3D;
    m_nsSolver->setGridDimensions(64,64,64);
    m_firstTime = false;
  }

  adjustAgeAndParticles(timestep);
  // copy from device to host
  cutilSafeCall(cudaMemcpy(m_hostPositionAge,m_devicePositionAge,sizeof(float4)*m_numParticles,cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(m_hostXVelocities,m_deviceXVelocities,sizeof(float)*m_numParticles,cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(m_hostYVelocities,m_deviceYVelocities,sizeof(float)*m_numParticles,cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(m_hostZVelocities,m_deviceZVelocities,sizeof(float)*m_numParticles,cudaMemcpyDeviceToHost));
  // run solver
  m_nsSolver->setParticles(m_hostPositionAge,m_hostXVelocities,m_hostYVelocities,m_hostZVelocities,m_numParticles);
  m_nsSolver->run(timestep);
  // copy back to device
  cutilSafeCall(cudaMemcpy(m_devicePositionAge,m_hostPositionAge,sizeof(float4)*m_numParticles,cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(m_deviceXVelocities,m_hostXVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(m_deviceYVelocities,m_hostYVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(m_deviceZVelocities,m_hostZVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice));
}

__global__ void adjustAgeAndMarkForRemoval(float4* atts, int* forRemoval, int numElements, float dt)
{
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= numElements)
    return;
  float4 val = atts[index];
  val = val - make_float4(dt,dt,dt,dt);
  val.x = max(val.x, 0.f); val.y = max(val.x, 0.f);
  val.z = max(val.x, 0.f); val.w = max(val.x, 0.f);
  atts[index] = val;
}

void CoarseParticleEngine::adjustAgeAndParticles(float dt)
{
  int blockSize = 512;
  int gridSize = (m_numParticles + blockSize - 1) / blockSize;
  float timeFactor = 0.03f;
  adjustAgeAndMarkForRemoval<<<gridSize,blockSize>>>(m_deviceFuelRadiusMassImpulse, m_particlesToRemove, m_numParticles, dt*timeFactor);
} 



static GLfloat quad[3] = { 1.0, 0.0, 1/60.0 };
void CoarseParticleEngine::render()
{
  glColor3f(1,1,1);
  glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, quad);
  glPointSize(8.0);
  glBindBuffer(GL_ARRAY_BUFFER, m_positionsAgeVBO);
  glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, m_numParticles);
  glDisableClientState(GL_VERTEX_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void CoarseParticleEngine::binParticles()
{
}

void CoarseParticleEngine::enforceDivergence()
{
}

void CoarseParticleEngine::projectToSlices()
{
}

void CoarseParticleEngine::resetParticles()
{
  m_numParticles = 0;
}

void CoarseParticleEngine::initializeParticles()
{
  // initialize host particle vectors
  m_hostPositionAge = (float4*) malloc(sizeof(float4)*m_maxNumParticles);
  m_hostFuelRadiusMassImpulse = (float4*) malloc(sizeof(float4)*m_maxNumParticles);
  m_hostXVelocities = (float*) malloc(sizeof(float)*m_maxNumParticles);
  m_hostYVelocities = (float*) malloc(sizeof(float)*m_maxNumParticles);
  m_hostZVelocities = (float*) malloc(sizeof(float)*m_maxNumParticles);
  
  // initialize device particle vectors
  cutilSafeCall(cudaMalloc((void**)&m_deviceFuelRadiusMassImpulse,sizeof(float4)*m_maxNumParticles));
  cutilSafeCall(cudaMalloc((void**)&m_deviceXVelocities,sizeof(float)*m_maxNumParticles));
  cutilSafeCall(cudaMalloc((void**)&m_deviceYVelocities,sizeof(float)*m_maxNumParticles));
  cutilSafeCall(cudaMalloc((void**)&m_deviceZVelocities,sizeof(float)*m_maxNumParticles));
  cutilSafeCall(cudaMalloc((void**)&m_particlesToRemove,sizeof(int)*m_maxNumParticles));

  // Create buffer object for position array and register it with CUDA
  glGenBuffers(1, &m_positionsAgeVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_positionsAgeVBO);
  unsigned int size = m_maxNumParticles * sizeof(float4);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_positionsVBO_CUDA, m_positionsAgeVBO, cudaGraphicsMapFlagsWriteDiscard));

  enableCUDAVbo();
  // update particle iterator struct (for passing around)
  m_particleStructItrBegin.posAge = m_devicePositionAge;
  m_particleStructItrBegin.atts = m_deviceFuelRadiusMassImpulse;
  m_particleStructItrBegin.velX = m_deviceXVelocities;
  m_particleStructItrBegin.velY = m_deviceYVelocities;
  m_particleStructItrBegin.velZ = m_deviceZVelocities;
  disableCUDAVbo();
}

void CoarseParticleEngine::addParticle(float3 position, float3 velocity, float fuel, 
                                       float radius, float lifetime, float mass, float impulse)
{
  if (m_numParticles >= m_maxNumParticles)
    return;

  m_hostPositionAge[m_numParticles] = make_float4(position.x,position.y,position.z,lifetime);
  m_hostFuelRadiusMassImpulse[m_numParticles] = make_float4(fuel,radius,mass,impulse);
  m_hostXVelocities[m_numParticles] = velocity.x;
  m_hostYVelocities[m_numParticles] = velocity.y;
  m_hostZVelocities[m_numParticles] = velocity.z;
  m_numParticles++;
}

void CoarseParticleEngine::addRandomParticle(float2 xBounds, float2 yBounds, float2 zBounds, int numParticles)
{
  float velMultiplier = .001f;
  for (int i = 0; i < numParticles; i++)
  {
    float3 newPosition = make_float3(randomFloatInRange(xBounds.x, xBounds.y), 
                                     randomFloatInRange(yBounds.x, yBounds.y), 
                                     randomFloatInRange(zBounds.x, zBounds.y));
    float3 newVelocity = make_float3(randomFloatInRange(0.f,10.f)*velMultiplier,
                                     randomFloatInRange(-1.f,1.f)*velMultiplier,
                                     randomFloatInRange(-1.f,1.f)*velMultiplier);
    // same attributes:
    //float randomVal = 1.f;//randomNormalizedFloat();
    //addParticle(newPosition, newVelocity, randomVal, 
    //            randomVal, randomVal, 
    //            randomVal, randomVal);

    // pos, vel, fuel, rad, lifetime, mass, impulse
    addParticle(newPosition, newVelocity, randomFloatInRange(0.4,1.0), 
                randomFloatInRange(0.0,1.0), randomFloatInRange(1.0, 3.0), 
                randomFloatInRange(0.0, 0.4), randomFloatInRange(0.5, 0.7));
  }
}

void CoarseParticleEngine::flushParticles()
{
  // get OpenGL position pointer and convert to CUDA pointer
  enableCUDAVbo();

  // make sure device vectors match host vectors
  cutilSafeCall(cudaMemcpy(m_devicePositionAge,m_hostPositionAge,sizeof(float4)*m_numParticles,cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(m_deviceFuelRadiusMassImpulse,m_hostFuelRadiusMassImpulse,sizeof(float4)*m_numParticles,cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(m_deviceXVelocities,m_hostXVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(m_deviceYVelocities,m_hostYVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(m_deviceZVelocities,m_hostZVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice));

  disableCUDAVbo();
}

void CoarseParticleEngine::enableCUDAVbo()
{
  cutilSafeCall(cudaGraphicsMapResources(1, &m_positionsVBO_CUDA, 0));
  size_t numBytes = m_maxNumParticles * sizeof(float4);
  cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&m_devicePositionAge, &numBytes, m_positionsVBO_CUDA));
}
 
void CoarseParticleEngine::disableCUDAVbo()
{
  cutilSafeCall(cudaGraphicsUnmapResources(1, &m_positionsVBO_CUDA, 0));
}
