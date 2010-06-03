#include "DensityTemperatureVolume.h"
#include <stdio.h>
#include <math.h>

DensityTemperatureVolume::DensityTemperatureVolume(const char* fileName)
{
}

DensityTemperatureVolume::~DensityTemperatureVolume()
{
  delete[] m_density;
  delete[] m_temperature;
}

float DensityTemperatureVolume::getDensityAt(float x, float y, float z)
{
  return trilinearlyInterpolate(m_density,x,y,z);
}

float DensityTemperatureVolume::getTempAt(float x, float y, float z)
{
  return trilinearlyInterpolate(m_temperature,x,y,z);
}

int DensityTemperatureVolume::getSizeInX()
{
  return m_dimX;
}

int DensityTemperatureVolume::getSizeInY()
{
  return m_dimY;
}

int DensityTemperatureVolume::getSizeInZ()
{
  return m_dimZ;
}

float DensityTemperatureVolume::trilinearlyInterpolate(const float* field, float x, float y, float z)
{
  if (x > m_dimX-1.f || x < 0.f || y > m_dimY-1.f || y < 0.f || z > m_dimZ-1.f || z < 0.f)
  {
    printf("invalid coordinate (%f, %f, %f) in range ([0,%i], [0,%i], [0,%i])\n",x,y,z,m_dimX,m_dimY,m_dimZ);
    return 0.f;
  }
  float floorX=floor(x), floorY=floor(y), floorZ=floor(z);
  float ceilX=ceil(x), ceilY=ceil(y), ceilZ=ceil(z);
  float xRatio = x - floor(x);
  float yRatio = y - floor(y);
  float zRatio = z - floor(z);
//  return getRawValueAt(
//  Vxyz = 	V000 (1 - x) (1 - y) (1 - z) +
//V100 x (1 - y) (1 - z) +
//V010 (1 - x) y (1 - z) +
//V001 (1 - x) (1 - y) z +
//V101 x (1 - y) z +
//V011 (1 - x) y z +
//V110 x y (1 - z) +
//V111 x y z 
}

float DensityTemperatureVolume::getRawValueAt(const float* field, int x, int y, int z)
{
  return field[y*x*z + y*x + x];
}