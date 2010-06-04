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
@file Renderer.cu
@note This file implements the renderer used to combine multiple slices into a single fire image
*/

#include "Renderer.h"
#include "XMLParser.h"
#include <cutil_inline.h>
#include <cutil_math.h>
#include "GL/glew.h"
#include "GL/glut.h"

texture<float, 1, cudaReadModeElementType> colorTexref;
static cudaArray *colorArray = NULL;

Renderer::Renderer(SliceManager* sliceManager, const char* settingsFileName)
: m_sliceManager(sliceManager), m_isMotionBlurred(false)
{

  loadConstantsFromFile(settingsFileName);
  initializeSlabPointers();
  cudaMalloc((void**)&m_d_imageBuffer, sizeof(float4)*m_sliceManager->getDomainSize());
  m_h_imageBuffer = (float4*) malloc(sizeof(float4)*m_sliceManager->getDomainSize());
  setupTexture();
}

Renderer::Renderer(const char* settingsFileName)
: m_sliceManager(0), m_isMotionBlurred(false)
{
  loadConstantsFromFile(settingsFileName);
  m_d_imageBuffer = 0;
  XMLParser settingsFile(settingsFileName);
  setupTexture();
}

void Renderer::loadConstantsFromFile(const char* settingsFileName)
{
  XMLParser settingsFile(settingsFileName);
  settingsFile.getFloat("textureDensityInfluence",&m_texDensInfluence);
  settingsFile.getFloat("textureTemperatureInfluence",&m_texTempInfluence);
  settingsFile.getFloat("densityAlphaExponent",&m_densityAlphaExp);
  settingsFile.getFloat("densityFactor",&m_maxDensity);
  settingsFile.getFloat("densityInvFactor",&m_densityInvFactor);
  settingsFile.setNewRoot("boundingBox");
  float zRange[2];
  settingsFile.getFloat2("zRange",zRange);
  m_sliceSpacing = (zRange[1] - zRange[0]) / m_sliceManager->getNumSlices();
  settingsFile.resetRoot();
}


Renderer::~Renderer()
{
  deleteTexture();
  cudaFree(m_d_imageBuffer);
  delete m_h_imageBuffer;
}

void Renderer::setupTexture() {
    // Wrap mode appears to be the new default
    colorTexref.filterMode = cudaFilterModeLinear;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaMallocArray(&colorArray, &desc, 12, 1);
    cudaMemcpyToArray(colorArray, 0, 0, colorTempsGreen, sizeof(float)*12, cudaMemcpyHostToDevice);
    cutilCheckMsg("color array texture failed");
}

void Renderer::bindTexture() {
    cudaBindTextureToArray(colorTexref, colorArray);
    cutilCheckMsg("cudaBindTexture failed");
}

void Renderer::unbindTexture() {
    cudaUnbindTexture(colorTexref);
}

void Renderer::deleteTexture() {
    cudaFreeArray(colorArray);
}

// 128 slices each including 3 scalar fields
#define MAX_SLICE_POINTERS (128*3)

void Renderer::initializeSlabPointers()
{
  cudaMalloc((void**)&m_d_slabPointers, sizeof(float*)*MAX_SLICE_POINTERS);
  int numSlices = m_sliceManager->getNumSlices();
  int numSlabs = 3;
  float** slabs = (float**) malloc(sizeof(float*)*MAX_SLICE_POINTERS);
  for (int i = 0; i < numSlices; i++)
  {
    float* densitySlab = m_sliceManager->getDensitySlab(i);
    float* temperatureSlab = m_sliceManager->getTemperatureSlab(i);
    float* textureSlab = m_sliceManager->getTextureSlab(i);
    slabs[numSlabs*i] = densitySlab;
    slabs[numSlabs*i+1] = temperatureSlab;
    slabs[numSlabs*i+2] = textureSlab;
  }
  cudaMemcpy(m_d_slabPointers, slabs, sizeof(float*)*MAX_SLICE_POINTERS, cudaMemcpyHostToDevice);
  delete slabs;
}

void Renderer::printConstants()
{
  printf("\n    CONSTANTS:\n");
  printf("texture density influence: %f\n",m_texDensInfluence);
  printf("texture temperature influence: %f\n",m_texTempInfluence);
  printf("density alpha exponent: %f\n",m_densityAlphaExp);
  printf("density inverse factor: %f\n",m_densityInvFactor);
}

void Renderer::increaseTexDensInfluence()
{
  m_texDensInfluence += 0.0015f;
}

void Renderer::increaseTexTempInfluence()
{
  m_texTempInfluence += 0.0015f;
}

void Renderer::increaseDensityAlphaExp()
{
  m_densityAlphaExp += .015f;
}

void Renderer::increaseDensityInv()
{
  m_densityInvFactor += .02f;
}

void Renderer::decreaseTexDensInfluence()
{
  m_texDensInfluence -= 0.0015f;
}

void Renderer::decreaseTexTempInfluence()
{
  m_texTempInfluence -= 0.0015f;
}

void Renderer::decreaseDensityAlphaExp()
{
  m_densityAlphaExp -= .015f;
}

void Renderer::decreaseDensityInv()
{
  m_densityInvFactor -= .02f;
}

__device__ bool validTemperature(float temperature)
{
  if (temperature < 1000.f || temperature > 2100.f)
    return false;
  return true;
}

__device__ float temperatureToGreenColor(float temperature)
{
  // convert temperature to index
  temperature -= 1000.f;
  temperature /= 100.f;
  // linearly interpolate color values
  return tex1D(colorTexref, temperature);
}

__device__ float temperaturePower(float temperature)
{
  if (temperature > 1700.f)
    return 1.f;
  temperature -= 1700.f;
  temperature /= -700.f;
  temperature = 1.f - temperature;
  temperature = powf(temperature, 3.f);
  return clamp(temperature, 0.f, 1.f);
}

#define RENDER_SHARED_MEM_SIZE (MAX_SLICE_POINTERS+12)
__global__ void renderSingleKernel(float4* frameBuffer, float** slabsAndColors, int bufferSize, int sliceToDisplay, int imageDim,
                                   float maxDensity, float textureDensityFactor, float textureTemperatureFactor, float densityExponent, float sliceSpacing,
                                   float densityInvFactor)
{
  __shared__ float* slabPointers[RENDER_SHARED_MEM_SIZE];
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= bufferSize)
    return;
  // load slabs into shared memory
  if (threadIdx.x < RENDER_SHARED_MEM_SIZE)
    slabPointers[threadIdx.x] = slabsAndColors[threadIdx.x];
  __syncthreads();
  float4 outputVal = make_float4(0,0,0,0);
  for (int i = 0; i < 1; i++)
  {
    float* densitySlab = slabPointers[3*sliceToDisplay];
    float* temperatureSlab = slabPointers[3*sliceToDisplay+1];
    float* textureSlab = slabPointers[3*sliceToDisplay+2];

    float currentDensity = densitySlab[index] /maxDensity;
    float currentTemperature = temperatureSlab[index];
    float currentTexture = textureSlab[index];
    float outputAlpha = currentDensity * (currentTexture * -textureDensityFactor + 1.f);
    outputAlpha *= (sliceSpacing / densityInvFactor);
    outputAlpha = powf(outputAlpha, densityExponent);
    if (validTemperature(currentTemperature))
    {
      currentTemperature = currentTemperature * (currentTexture * textureTemperatureFactor + 1.f);
      float greenVal = temperatureToGreenColor(currentTemperature);
      float tempPower = temperaturePower(currentTemperature);
      outputVal += make_float4(255.f * outputAlpha * tempPower, greenVal * outputAlpha * tempPower, 0.f, outputAlpha * tempPower);
    }
    else
      outputVal += make_float4(0,0,0,0);
  }
  outputVal.x = max(min(outputVal.x,1.f),0.f); outputVal.y = max(min(outputVal.y,1.f),0.f);
  outputVal.z = max(min(outputVal.z,1.f),0.f); outputVal.w = max(min(outputVal.w,1.f),0.f);
  frameBuffer[index] = outputVal;
}

void Renderer::renderSingleSlice(int sliceToDisplay)
{
  if (sliceToDisplay < 0 || sliceToDisplay >= m_sliceManager->getNumSlices())
    return;
  bindTexture();
  int blockSize = 512;
  int domainSize = m_sliceManager->getDomainSize();
  int gridSize = (domainSize + blockSize - 1) / blockSize;
  int imageDim = m_sliceManager->getImageDim();
  renderSingleKernel<<<gridSize,blockSize>>>(m_d_imageBuffer, m_d_slabPointers, domainSize, sliceToDisplay, imageDim,
    m_maxDensity, m_texDensInfluence, m_texTempInfluence, m_densityAlphaExp, m_sliceSpacing, m_densityInvFactor);
  cudaMemcpy(m_h_imageBuffer, m_d_imageBuffer,sizeof(float4)*domainSize, cudaMemcpyDeviceToHost);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE);
  glWindowPos2i(0,0);
  glDrawPixels(imageDim,imageDim,GL_RGBA,GL_FLOAT,m_h_imageBuffer);
  glDisable(GL_BLEND);
  unbindTexture();
}

__global__ void renderCompositeKernel(float4* frameBuffer, float** slabsAndColors, int bufferSize, int numSlices, int imageDim,
                                   float maxDensity, float textureDensityFactor, float textureTemperatureFactor, float densityExponent, float sliceSpacing,
                                   float densityInvFactor)
{
  __shared__ float* slabPointers[RENDER_SHARED_MEM_SIZE];
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= bufferSize)
    return;
  // load slabs into shared memory
  if (threadIdx.x < RENDER_SHARED_MEM_SIZE)
    slabPointers[threadIdx.x] = slabsAndColors[threadIdx.x];
  __syncthreads();
  float4 outputVal = make_float4(0,0,0,0);
  for (int i = 0; i < (numSlices-1); i++)
  {
    float* densitySlab = slabPointers[3*i];
    float* temperatureSlab = slabPointers[3*i+1];
    float* textureSlab = slabPointers[3*i+2];

    float currentDensity = densitySlab[index] /maxDensity;
    float currentTemperature = temperatureSlab[index];
    float currentTexture = textureSlab[index];
    float outputAlpha = currentDensity * (currentTexture * -textureDensityFactor + 1.f);
    outputAlpha *= (sliceSpacing / densityInvFactor);
    outputAlpha = powf(outputAlpha, densityExponent);
    if (validTemperature(currentTemperature))
    {
      currentTemperature = currentTemperature * (currentTexture * textureTemperatureFactor + 1.f);
      float greenVal = temperatureToGreenColor(currentTemperature);
      float tempPower = temperaturePower(currentTemperature);
      outputVal += make_float4(255.f * outputAlpha * tempPower, greenVal * outputAlpha * tempPower, 0.f, outputAlpha * tempPower);
    }
    else
      outputVal += make_float4(0,0,0,0);
  }
  outputVal.x = max(min(outputVal.x,1.f),0.f); outputVal.y = max(min(outputVal.y,1.f),0.f);
  outputVal.z = max(min(outputVal.z,1.f),0.f); outputVal.w = max(min(outputVal.w,1.f),0.f);
  frameBuffer[index] = outputVal;
}

void Renderer::renderComposite()
{
  bindTexture();
  int blockSize = 512;
  int domainSize = m_sliceManager->getDomainSize();
  int gridSize = (domainSize + blockSize - 1) / blockSize;
  int imageDim = m_sliceManager->getImageDim();
  int numSlices = m_sliceManager->getNumSlices();
  renderCompositeKernel<<<gridSize,blockSize>>>(m_d_imageBuffer, m_d_slabPointers, domainSize, numSlices, imageDim,
    m_maxDensity, m_texDensInfluence, m_texTempInfluence, m_densityAlphaExp, m_sliceSpacing, m_densityInvFactor);
  if (m_isMotionBlurred)
    performMotionBlur();
  else
  {
    cudaMemcpy(m_h_imageBuffer, m_d_imageBuffer,sizeof(float4)*domainSize, cudaMemcpyDeviceToHost);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    glWindowPos2i(0,0);
    glDrawPixels(imageDim,imageDim,GL_RGBA,GL_FLOAT,m_h_imageBuffer);
    glDisable(GL_BLEND);
  }
   unbindTexture();
}

__global__ void advectRenderedImage(float4* imageBuffer, float** slabsAndColors, int domainSize, int imageDim)
{
  __shared__ float* slabPointers[RENDER_SHARED_MEM_SIZE];
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= domainSize)
    return;
  // load slabs into shared memory
  if (threadIdx.x < RENDER_SHARED_MEM_SIZE)
    slabPointers[threadIdx.x] = slabsAndColors[threadIdx.x];
  __syncthreads();
  float4 outputVal = make_float4(0,0,0,0);

  imageBuffer[index] = outputVal;
}

void Renderer::performMotionBlur()
{
  glClear(GL_ACCUM_BUFFER_BIT);
  int numMotionBlurIterations = 10;
  int blockSize = 512;
  int domainSize = m_sliceManager->getDomainSize();
  int imageDim = m_sliceManager->getImageDim();
  int gridSize = (domainSize + blockSize - 1) / blockSize;
  for (int i = 0; i < numMotionBlurIterations; i++) {
    advectRenderedImage<<<gridSize,blockSize>>>(m_d_imageBuffer, m_d_slabPointers, domainSize, imageDim);
    cudaMemcpy(m_h_imageBuffer, m_d_imageBuffer,sizeof(float4)*domainSize, cudaMemcpyDeviceToHost);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    glWindowPos2i(0,0);
    glDrawPixels(imageDim,imageDim,GL_RGBA,GL_FLOAT,m_h_imageBuffer);
    glDisable(GL_BLEND);
    glAccum(GL_ACCUM, 1.f / numMotionBlurIterations);
  }
  glAccum(GL_RETURN, 1.f);

}