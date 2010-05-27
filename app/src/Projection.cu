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
@file Projection.cu
*/

#include "Projection.h"
#include <cutil_math.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/unique.h>
#include <thrust/scan.h>


////////////////// PROJECTION //////////////////////

Projection::Projection(float3 gridCenter, float3 gridDimensions, float projectionDepth, int2 slicePixelDimensions, int maxNumParticles, int numJitter)
: m_gridCenter(gridCenter), m_gridDims(gridDimensions), m_projectionDepth(projectionDepth), m_slicePixelDims(slicePixelDimensions), 
m_numParticles(0), m_outputMassSlice(0), m_outputFuelSlice(0), m_outputVelocitySlice(0), m_maxNumParticles(maxNumParticles), m_numJitter(numJitter)
{
  cudaMalloc((void**)&m_binIndex,sizeof(int)*m_maxNumParticles*m_numJitter);
  cudaMalloc((void**)&m_particleIndex,sizeof(int)*m_maxNumParticles*m_numJitter);
  cudaMalloc((void**)&m_binValue,sizeof(int)*m_maxNumParticles*m_numJitter);

  cudaMalloc((void**)&m_massFuelVelocity,sizeof(float4)*m_maxNumParticles*m_numJitter);
  cudaMalloc((void**)&m_summedMassFuelVelocity,sizeof(float4)*m_maxNumParticles*m_numJitter);
}

void Projection::setParticles(ParticleItrStruct particlesBegin, int numParticles)
{
  m_particlesBegin = particlesBegin;
  m_numParticles = numParticles;
    
}

//////////// ORTHOGRAPHIC PROJECTION //////////////////////

OrthographicProjection::OrthographicProjection(float3 gridCenter, float3 gridDimensions, float projectionDepth, 
                                               int2 slicePixelDimensions, float2 sliceWorldDimensions, int maxNumParticles, int numJitter)
: Projection(gridCenter,gridDimensions,projectionDepth,slicePixelDimensions,maxNumParticles,numJitter), m_ZIntercept(-1.f), m_sliceWorldDims(sliceWorldDimensions)
{
}

//void OrthographicProjection::setSliceInformation(float zIntercept, float* outputMassSlice, float* m_outputFuelSlice, float2* outputVelocitySlice)
//{
//  m_ZIntercept = zIntercept;
//  m_outputMassSlice = outputMassSlice;
//  m_outputFuelSlice = m_outputFuelSlice;
//  m_outputVelocitySlice = outputVelocitySlice;
//}



#define M_E 2.7182818284590452f
__device__ float particleWeight(float projectionDistance, float sliceSpacing)
{
  float exponent = (-projectionDistance*projectionDistance)/(16*sliceSpacing*sliceSpacing);
  return powf(M_E,exponent);
}

__device__ int worldToPixelIndex(float4 position, float2 xyLowerBound, float2 xyUpperBound, int2 imageDims)
{
  int pixelIndex;
  if (position.x < xyLowerBound.x) position.x = xyLowerBound.x;
  else if (position.x >= xyUpperBound.x) position.x = xyUpperBound.x -1;

  if (position.y < xyLowerBound.y) position.y = xyLowerBound.y;
  else if (position.y >= xyUpperBound.y) position.y = xyUpperBound.y -1;

  float xRatio = (position.x - xyLowerBound.x) / (xyUpperBound.x-xyLowerBound.x);
  float yRatio = 1.f + (position.y - xyUpperBound.y) / (xyUpperBound.y-xyLowerBound.y);
  int xVal = xRatio * imageDims.x;
  int yVal = yRatio * imageDims.y;
  pixelIndex = yVal*imageDims.x+xVal;

  return pixelIndex;

}

__global__ void generateKeys(ParticleItrStruct particles, int* keys, int* particleIndices, int numParticles, float2 xyLowerBound, float2 xyUpperBound, int2 imageDims, int numJitters)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numParticles)
    return;
  float4 particlePosition = particles.posAge[index];
  float4 particleVelocity = make_float4(particles.velX[index], particles.velY[index], particles.velZ[index], 0);
  for (int i=0; i < numJitters; i++)
  {
    // compute which row the pixel lies in
    int pixelIndex = worldToPixelIndex(particlePosition,xyLowerBound,xyUpperBound,imageDims);
    if (pixelIndex == -1)
      pixelIndex = 0;
    // write out the current particle index and the bin index
    keys[index + numParticles*i] = pixelIndex;
    particleIndices[index + numParticles*i] = index;
    particlePosition += particleVelocity * 0.0006f;
  }
}

__global__ void transformParticleIndices(ParticleItrStruct particles, int* particleIndices, float4* massFuelVelocity, 
                                         int numElements, float sliceDepth, float zIntercept)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElements)
    return;
  int particleIndex = particleIndices[index];
  // get particle values
  float4 particlePositionAge = particles.posAge[particleIndex];
  float4 particleAttributes = particles.atts[particleIndex];
  float2 particleVelocity = make_float2(particles.velX[particleIndex], particles.velY[particleIndex]);
  // get weight of particle based on gaussian function particleWeight
  float weight = particleWeight(fabs(particlePositionAge.z - zIntercept), sliceDepth);
  // write values out
  massFuelVelocity[index] = make_float4(particleAttributes.z * weight,
    particleAttributes.x * weight,
    particleVelocity.x * particleAttributes.w * weight,
    particleVelocity.y * particleAttributes.w * weight);
}

__global__ void writeParticleSumOut(float4* massFuelVelocitySum, int* binIndex, float* mass, float* fuel, float2* velocity, int numElements)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElements)
    return;
  int bin = binIndex[index];
  float4 massFuelVel = massFuelVelocitySum[index];
  mass[bin] = massFuelVel.x;
  fuel[bin] = massFuelVel.y;
  velocity[bin] = make_float2(massFuelVel.z, massFuelVel.w);
}

struct float4Plus 
  : public thrust::binary_function<float,float,float>
{
 __host__ __device__ float4 operator()(const float4 &lhs, const float4 &rhs) const {return lhs + rhs;}
};

void OrthographicProjection::testProjection(float zIntercept, float* outputMassSlice, float* outputFuelSlice, float2* outputVelocitySlice)
{
  if (!outputMassSlice || !outputFuelSlice || !outputVelocitySlice)
    exit(1);
  int blockSize = 512;
  int gridSize = (m_numParticles + blockSize - 1) / blockSize;
  float2 xyLowerBound = make_float2(m_gridCenter.x - m_gridDims.x/2.f, m_gridCenter.y - m_gridDims.y/2.f);
  float2 xyUpperBound = make_float2(m_gridCenter.x + m_gridDims.x/2.f, m_gridCenter.y + m_gridDims.y/2.f);
  int numJitteredParticles = m_numParticles*m_numJitter;
  
  // generate keys (the row of the particle) for each particle and all its jitters
  generateKeys<<<gridSize,blockSize>>>(m_particlesBegin, m_binIndex, m_particleIndex, m_numParticles, xyLowerBound, xyUpperBound, m_slicePixelDims, m_numJitter);
  
  // sort by bin so adjacent bins are all next to each other
  thrust::device_ptr<int> devPtrBinIndex(m_binIndex);
  thrust::device_ptr<int> devPtrParticleIndex(m_particleIndex);
  thrust::sort_by_key(devPtrBinIndex, devPtrBinIndex+numJitteredParticles, devPtrParticleIndex);
  // transform indices to values to allow reduce_by_key
  int blockSize2 = 512;
  int gridSize2 = (numJitteredParticles + blockSize2 - 1) / blockSize2;
  transformParticleIndices<<<gridSize2,blockSize2>>>(m_particlesBegin,m_particleIndex,m_massFuelVelocity,
    numJitteredParticles,m_projectionDepth,zIntercept);
  // reduce each bin's particles to their sum
  thrust::device_ptr<int> devPtrBinValue(m_binValue);
  thrust::device_ptr<float4> devPtrMassFuelVel(m_massFuelVelocity);
  thrust::device_ptr<float4> devPtrSumMassFuelVel(m_summedMassFuelVelocity);
  thrust::pair<thrust::device_ptr<int>,thrust::device_ptr<float4> > new_end;
  thrust::equal_to<int> binary_pred;
  float4Plus binary_op;
  new_end = thrust::reduce_by_key(devPtrBinIndex, devPtrBinIndex+numJitteredParticles, devPtrMassFuelVel, 
    devPtrBinValue, devPtrSumMassFuelVel,binary_pred,binary_op);

  // write each bin out to image
  int numBinsUsed = (new_end.first.get() - devPtrBinValue.get());
  int blockSize3 = 512;
  int gridSize3 = (numBinsUsed + blockSize3 - 1) / blockSize3;
  writeParticleSumOut<<<gridSize3,blockSize3>>>(m_summedMassFuelVelocity, m_binValue, outputMassSlice, outputFuelSlice, outputVelocitySlice, numBinsUsed);
}

void OrthographicProjection::naiveProjection()
{
  //if (m_outputSlice == 0)
  //  exit(1);
  //int blockSize = 512;
  //int gridSize = (m_numParticles + blockSize - 1) / blockSize;
  //float2 xyLowerBound = make_float2(m_gridCenter.x - m_gridDims.x/2.f, m_gridCenter.y - m_gridDims.y/2.f);
  //float2 xyUpperBound = make_float2(m_gridCenter.x + m_gridDims.x/2.f, m_gridCenter.y + m_gridDims.y/2.f);

  //orthProjection<<<gridSize,blockSize>>>(m_particlesBegin,m_outputSlice,m_ZIntercept,m_projectionDepth,
  //                                       m_numParticles,xyLowerBound,xyUpperBound,m_slicePixelDims);
  //orthVelocityProjection<<<gridSize,blockSize>>>(m_particlesBegin,m_outputVelocitySlice,m_ZIntercept,m_projectionDepth,
  //                                       m_numParticles,xyLowerBound,xyUpperBound,m_slicePixelDims);
}

void OrthographicProjection::execute(float zIntercept, float* outputMassSlice, float* outputFuelSlice, float2* outputVelocitySlice)
{
  //naiveProjection();
  testProjection(zIntercept, outputMassSlice, outputFuelSlice, outputVelocitySlice);
}





