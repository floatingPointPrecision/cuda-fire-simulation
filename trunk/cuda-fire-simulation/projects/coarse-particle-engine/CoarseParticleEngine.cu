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
#include <cuda_gl_interop.h>
#include <cutil_gl_inline.h>
#include "thrust/copy.h"
#include "thrust/transform.h"
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

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
    m_nsSolver->setParticles(m_hostPositionAge,m_hostXVelocities,m_hostYVelocities,m_hostZVelocities,m_numParticles);
    m_firstTime = false;
  }
  // test ns solver
  // copy from device to host
  CPUTimer timer;
  timer.start();
  cudaMemcpy(m_hostPositionAge,m_devicePositionAge,sizeof(float4)*m_numParticles,cudaMemcpyDeviceToHost);
  cudaMemcpy(m_hostXVelocities,m_deviceXVelocities,sizeof(float)*m_numParticles,cudaMemcpyDeviceToHost);
  cudaMemcpy(m_hostYVelocities,m_deviceYVelocities,sizeof(float)*m_numParticles,cudaMemcpyDeviceToHost);
  cudaMemcpy(m_hostZVelocities,m_deviceZVelocities,sizeof(float)*m_numParticles,cudaMemcpyDeviceToHost);
  // run solver
  CPUTimer timer2;
  timer2.start();
  m_nsSolver->run();
  timer2.stop();
  printf("3D NS solver: %f\n", timer2.elapsed_sec());
  // copy back to device
  cudaMemcpy(m_devicePositionAge,m_hostPositionAge,sizeof(float4)*m_numParticles,cudaMemcpyHostToDevice);
  cudaMemcpy(m_deviceXVelocities,m_hostXVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice);
  cudaMemcpy(m_deviceYVelocities,m_hostYVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice);
  cudaMemcpy(m_deviceZVelocities,m_hostZVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice);

  timer.stop();
  printf("3D NS complete: %f\n", timer.elapsed_sec());
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
  cudaMalloc((void**)&m_deviceFuelRadiusMassImpulse,sizeof(float4)*m_maxNumParticles);
  cudaMalloc((void**)&m_deviceXVelocities,sizeof(float)*m_maxNumParticles);
  cudaMalloc((void**)&m_deviceYVelocities,sizeof(float)*m_maxNumParticles);
  cudaMalloc((void**)&m_deviceZVelocities,sizeof(float)*m_maxNumParticles);

  // Create buffer object for position array and register it with CUDA
  glGenBuffers(1, &m_positionsAgeVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_positionsAgeVBO);
  unsigned int size = m_maxNumParticles * sizeof(float4);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGLRegisterBufferObject(m_positionsAgeVBO);
  enableCUDAVbo();

  // update particle iterator struct (for passing around)
  m_particleStructItrBegin.posAge = m_devicePositionAge;
  m_particleStructItrBegin.atts = m_deviceFuelRadiusMassImpulse;
  m_particleStructItrBegin.velX = m_deviceXVelocities;
  m_particleStructItrBegin.velY = m_deviceYVelocities;
  m_particleStructItrBegin.velZ = m_deviceZVelocities;

  //m_nsSolver->setGridDimensions(64,64,64);
  //m_nsSolver->setParticles(m_hostPositionAge,m_hostXVelocities,m_hostYVelocities,m_hostZVelocities,m_numParticles);

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
    addParticle(newPosition, newVelocity, randomNormalizedFloat(), 
                randomNormalizedFloat(), randomNormalizedFloat(), 
                randomNormalizedFloat(), randomNormalizedFloat());
  }
}

void CoarseParticleEngine::flushParticles()
{
  // get OpenGL position pointer and convert to CUDA pointer
  enableCUDAVbo();

  // make sure device vectors match host vectors
  cudaMemcpy(m_devicePositionAge,m_hostPositionAge,sizeof(float4)*m_numParticles,cudaMemcpyHostToDevice);
  cudaMemcpy(m_deviceFuelRadiusMassImpulse,m_hostFuelRadiusMassImpulse,sizeof(float4)*m_numParticles,cudaMemcpyHostToDevice);
  cudaMemcpy(m_deviceXVelocities,m_hostXVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice);
  cudaMemcpy(m_deviceYVelocities,m_hostYVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice);
  cudaMemcpy(m_deviceZVelocities,m_hostZVelocities,sizeof(float)*m_numParticles,cudaMemcpyHostToDevice);

  disableCUDAVbo();
}

void CoarseParticleEngine::enableCUDAVbo()
{
  cudaGLMapBufferObject((void**)&m_devicePositionAge, m_positionsAgeVBO);
}
 
void CoarseParticleEngine::disableCUDAVbo()
{
  cudaGLUnmapBufferObject(m_positionsAgeVBO);
}
