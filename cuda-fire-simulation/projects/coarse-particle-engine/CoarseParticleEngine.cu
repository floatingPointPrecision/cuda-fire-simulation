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
#include "Bitmap.hpp"
#include "projectTest.h"

using namespace cufire;

////////////////// CUDA CODE ////////////////////
__host__ __device__ 
ParticleTuple MakeParticleTuple(float4 positionAge, float4 fuelRadMassImpulse, float xVel, float yVel, float zVel)
{
  return thrust::make_tuple<float4,float4,float,float,float>(positionAge,fuelRadMassImpulse,xVel,yVel,zVel);
}

/**
  * update position
  * Updates the particle position
  */
struct UpdateParticlePosition : public thrust::unary_function<ParticleTuple,ParticleTuple>
{
  UpdateParticlePosition(float timeStep)
    : dT(timeStep)
  {}

  __host__ __device__
    ParticleTuple operator()(const ParticleTuple& a) const
  {
    float4 posAge = thrust::get<0>(a);
    float4 fuelRadMassImpulse = thrust::get<1>(a);
    float velX = thrust::get<2>(a);
    float velY = thrust::get<3>(a);
    float velZ = thrust::get<4>(a);
    posAge.x += velX * dT;
    posAge.y += velY * dT;
    posAge.z += velZ * dT;
    posAge.w -= dT;
    return MakeParticleTuple(posAge,
                             fuelRadMassImpulse,
                             velX,velY,velZ);
  }
  float dT;
};

////////////////// HOST CODE ////////////////////
CoarseParticleEngine::CoarseParticleEngine(int maxNumberParticles)
: m_maxNumParticles(maxNumberParticles), m_currentTime(0.f)
{
  initializeParticles();
  resetParticles();
}

CoarseParticleEngine::~CoarseParticleEngine()
{
  cudaGLUnregisterBufferObject(m_positionsAgeVBO);
  glDeleteBuffers(1, &m_positionsAgeVBO);
}

void CoarseParticleEngine::advanceSimulation(float timestep)
{
  m_currentTime += timestep;
  float4* glPosPointer;
  cudaGLMapBufferObject((void**)&glPosPointer, m_positionsAgeVBO);
  DevPtrFloat4 rawDevicePositionPtr(glPosPointer);
  // create hypothetical new velocities from artist-defined forces
  thrust::transform(m_particleItrBegin, m_particleItrBegin + m_numParticles, m_particleItrBegin, UpdateParticlePosition(1/60.f));
  // bin particles into grids for non-divergence calculation

  // solve for non-divergence

  // add non-divergence term back to hypothetical new velocities

  // update particle position according to new velocities

  // perform projection
  float3 gridCenter = make_float3(0,0,0);
  float3 gridDims = make_float3(8,8,8);
  float projectionDepth = 0.5;
  int2 slicePixelDims = make_int2(300,300);
  float2 sliceWorldDims = make_float2(8,8);
  int numSliceBytes = sizeof(float4)*slicePixelDims.x*slicePixelDims.y;
  float4* d_sliceOutput;
  float4* h_sliceOutput = (float4*) malloc(numSliceBytes);
  cudaMalloc((void**)&d_sliceOutput, numSliceBytes);
  cudaMemset(d_sliceOutput,0, numSliceBytes);
  OrthographicProjection proj(gridCenter,gridDims,projectionDepth,slicePixelDims,sliceWorldDims);
  proj.setSliceInformation(3.f, d_sliceOutput);
  proj.setParticles(m_particleStructItrBegin,m_numParticles);
  proj.execute();
  // copy output as image
  cudaMemcpy(h_sliceOutput, d_sliceOutput, numSliceBytes, cudaMemcpyDeviceToHost);
  BitmapWriter newBitmap(slicePixelDims.x,slicePixelDims.y);
  int x,y,width=slicePixelDims.x,height=slicePixelDims.y;
  for(x=0;x<width;++x)
  {
    for(y=0;y<height;++y)
    {
      float4 currentPixel = h_sliceOutput[y*width+x];
      newBitmap.setValue(x,y,char(currentPixel.x),char(currentPixel.y),char(currentPixel.z));
    }
  }
  newBitmap.flush("outputBitmap.bmp");
  cudaFree(d_sliceOutput);
  free(h_sliceOutput);

  // test ns solver
  ProjectTest newTest;
  newTest.run();

  cudaGLUnmapBufferObject(m_positionsAgeVBO);
}

static GLfloat quad[3] = { 1.0, 0.0, 1/60.0 };
void CoarseParticleEngine::render()
{
  glColor3f(1,0,0);
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
  m_hostPositionAge = thrust::host_vector<float4>(m_maxNumParticles);
  m_hostFuelRadiusMassImpulse = thrust::host_vector<float4>(m_maxNumParticles);
  m_hostXVelocities = thrust::host_vector<float>(m_maxNumParticles);
  m_hostYVelocities = thrust::host_vector<float>(m_maxNumParticles);
  m_hostZVelocities = thrust::host_vector<float>(m_maxNumParticles);
  
  // initialize device particle vectors
  m_deviceFuelRadiusMassImpulse = thrust::device_vector<float4>(m_maxNumParticles);
  m_deviceXVelocities = thrust::device_vector<float>(m_maxNumParticles);
  m_deviceYVelocities = thrust::device_vector<float>(m_maxNumParticles);
  m_deviceZVelocities = thrust::device_vector<float>(m_maxNumParticles);
  // Create buffer object for position array and register it with CUDA
  glGenBuffers(1, &m_positionsAgeVBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_positionsAgeVBO);
  unsigned int size = m_maxNumParticles * sizeof(float4);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGLRegisterBufferObject(m_positionsAgeVBO);

  //// create random number generator
  //// create a minstd_rand object to act as our source of randomness
  //thrust::minstd_rand rng;
  //// create a uniform_real_distribution to produce floats from [-7,13)
  //thrust::uniform_real_distribution<float> dist(-1.f,1.f);
  //for (int i = 0; i < 16; i++)
  //  std::cout << "random number: " << dist(rng) << std::endl;

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
  
  //std::cout << "particle added: position (" << position.x << "," << position.y << "," << position.z <<
  //          ")" << std::endl;
  m_numParticles++;
}

void CoarseParticleEngine::addRandomParticle(float2 xBounds, float2 yBounds, float2 zBounds, int numParticles)
{
  for (int i = 0; i < numParticles; i++)
  {
    float3 newPosition = make_float3(randomFloatInRange(xBounds.x, xBounds.y), 
                                     randomFloatInRange(yBounds.x, yBounds.y), 
                                     randomFloatInRange(zBounds.x, zBounds.y));
    float3 newVelocity = make_float3(randomFloatInRange(-1.f,1.f),
                                     randomFloatInRange(-1.f,1.f),
                                     randomFloatInRange(-1.f,1.f));
    addParticle(newPosition, newVelocity, randomNormalizedFloat(), 
                randomNormalizedFloat(), randomNormalizedFloat(), 
                randomNormalizedFloat(), randomNormalizedFloat());
  }
}

void CoarseParticleEngine::flushParticles()
{
  // get OpenGL position pointer and convert to device_ptr
  float4* glPositionAges;
  cudaGLMapBufferObject((void**)&glPositionAges, m_positionsAgeVBO);
  DevPtrFloat4 m_positionsAgeRaw(glPositionAges);

  // make sure device vectors match host vectors
  m_deviceFuelRadiusMassImpulse = m_hostFuelRadiusMassImpulse;
  m_deviceXVelocities = m_hostXVelocities;
  m_deviceYVelocities = m_hostYVelocities;
  m_deviceZVelocities = m_hostZVelocities;

  // update particle iterator struct (for passing around)
  m_particleStructItrBegin.posAge = glPositionAges;
  m_particleStructItrBegin.atts = m_deviceFuelRadiusMassImpulse.begin();
  m_particleStructItrBegin.velX = m_deviceXVelocities.begin();
  m_particleStructItrBegin.velY = m_deviceYVelocities.begin();
  m_particleStructItrBegin.velZ = m_deviceZVelocities.begin();

  m_particleStructItrEnd.posAge = glPositionAges + m_numParticles;
  m_particleStructItrEnd.atts = m_deviceFuelRadiusMassImpulse.begin() + m_numParticles;
  m_particleStructItrEnd.velX = m_deviceXVelocities.begin() + m_numParticles;
  m_particleStructItrEnd.velY = m_deviceYVelocities.begin() + m_numParticles;
  m_particleStructItrEnd.velZ = m_deviceZVelocities.begin() + m_numParticles;

  // update zip_iterators
  m_particleItrBegin = thrust::make_zip_iterator(make_tuple(m_positionsAgeRaw,
    m_deviceFuelRadiusMassImpulse.begin(),
    m_deviceXVelocities.begin(),
    m_deviceYVelocities.begin(),
    m_deviceZVelocities.begin()));
  m_particleItrEnd = thrust::make_zip_iterator(make_tuple(m_positionsAgeRaw + m_numParticles,
    m_deviceFuelRadiusMassImpulse.begin() + m_numParticles,
    m_deviceXVelocities.begin() + m_numParticles,
    m_deviceYVelocities.begin() + m_numParticles,
    m_deviceZVelocities.begin() + m_numParticles));

  // incredibly slow, need to figure out how to work with device_vector from OpenGL vbo
  thrust::copy(m_hostPositionAge.begin(), m_hostPositionAge.begin()+m_numParticles,m_positionsAgeRaw);
  //for (int i = 0; i < m_numParticles; i++)
  //  m_positionsAgeRaw[i] = m_hostPositionAge[i];

  cudaGLUnmapBufferObject(m_positionsAgeVBO);
}
