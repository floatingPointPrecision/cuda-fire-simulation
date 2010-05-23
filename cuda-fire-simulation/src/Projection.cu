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

////////////////// PROJECTION //////////////////////

Projection::Projection(float3 gridCenter, float3 gridDimensions, float projectionDepth, int2 slicePixelDimensions)
: m_gridCenter(gridCenter), m_gridDims(gridDimensions), m_projectionDepth(projectionDepth), m_slicePixelDims(slicePixelDimensions), m_numParticles(0), m_outputSlice(0)
{
}

//////////// ORTHOGRAPHIC PROJECTION //////////////////////

OrthographicProjection::OrthographicProjection(float3 gridCenter, float3 gridDimensions, float projectionDepth, int2 slicePixelDimensions, float2 sliceWorldDimensions)
: Projection(gridCenter,gridDimensions,projectionDepth,slicePixelDimensions), m_ZIntercept(-1.f), m_sliceWorldDims(sliceWorldDimensions)
{
}

void OrthographicProjection::setSliceInformation(float zIntercept, float4* outputSlice)
{
  m_ZIntercept = zIntercept;
  m_outputSlice = outputSlice;
}

#define M_E 2.7182818284590452f
__device__ float particleWeight(float projectionDistance, float sliceSpacing)
{
  float exponent = (-projectionDistance*projectionDistance)/(16*sliceSpacing*sliceSpacing);
  return powf(M_E,exponent);
}

__device__ int worldToPixelIndex(float4 position, float2 xyLowerBound, float2 xyUpperBound, int2 imageDims)
{
  int pixelIndex;
  if (position.x < xyLowerBound.x || position.x > xyUpperBound.x ||
      position.y < xyLowerBound.y || position.y > xyUpperBound.y)
    pixelIndex = -1;
  else
  {
    float xRatio = (position.x - xyLowerBound.x) / (xyUpperBound.x-xyLowerBound.x);
    float yRatio = 1.f + (position.y - xyUpperBound.y) / (xyUpperBound.y-xyLowerBound.y);
    int xVal = xRatio * imageDims.x;
    int yVal = yRatio * imageDims.y;
    pixelIndex = yVal*imageDims.x+xVal;
  }
  return pixelIndex;

} 
__global__ void orthProjection(ParticleItrStruct particles, float4* output, float zIntercept, float sliceDepth, 
                               int numParticles, float2 xyLowerBound, float2 xyUpperBound, int2 imageDims)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numParticles)
    return;
  float4 positionAge = particles.posAge[index];
  float4 particleVelocity = make_float4(particles.velX[index],particles.velY[index],particles.velZ[index],0);
  for (int i = 0; i < 10; i++)
  {
    int pixelIndex = worldToPixelIndex(positionAge,xyLowerBound,xyUpperBound,imageDims);
    float projectionDistance = fabs(positionAge.z - zIntercept);
    float projectionWeight = particleWeight(projectionDistance, sliceDepth);
    if (pixelIndex == -1)
      continue;
    projectionWeight *= 255.f;
    float4 oldPixelValue = output[pixelIndex];
    oldPixelValue += make_float4(projectionWeight,projectionWeight,projectionWeight,0);
    output[pixelIndex] = oldPixelValue;
    positionAge += particleVelocity*10;
  }
}

void OrthographicProjection::execute()
{
  if (m_outputSlice == 0)
    exit(1);
  int blockSize = 512;
  int gridSize = (m_numParticles + blockSize - 1) / blockSize;
  float2 xyLowerBound = make_float2(m_gridCenter.x - m_gridDims.x/2.f, m_gridCenter.y - m_gridDims.y/2.f);
  float2 xyUpperBound = make_float2(m_gridCenter.x + m_gridDims.x/2.f, m_gridCenter.y + m_gridDims.y/2.f);
  orthProjection<<<gridSize,blockSize>>>(m_particlesBegin,m_outputSlice,m_ZIntercept,m_projectionDepth,
                                         m_numParticles,xyLowerBound,xyUpperBound,m_slicePixelDims);
}





