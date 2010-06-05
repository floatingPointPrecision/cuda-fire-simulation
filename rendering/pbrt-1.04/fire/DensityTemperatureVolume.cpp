#include "DensityTemperatureVolume.h"
#include <stdio.h>
#include <math.h>

DensityTemperatureVolume::DensityTemperatureVolume(const char* fileName)
{
  FILE* inFile = fopen(fileName,"rb");
  if (!inFile)
  {
    printf("unable to open file %s\n",fileName);
    return;
  }
  // read in the header which includes the image space dimensions
  int header[3];
  int headerSize = sizeof(int)*3;
  fread(header,sizeof(int),3,inFile);  
  m_dimX = header[0];
  m_dimY = header[1];
  m_dimZ = header[2];
  int domainSize = m_dimX*m_dimY*m_dimZ;
  // allocate density and temperature accordingly
  m_density = new float[domainSize];
  m_temperature = new float[domainSize];
  // read in density
  fseek(inFile,headerSize,SEEK_SET);
  fread(m_density,sizeof(float),domainSize,inFile);
  // read in temperature
  fseek(inFile,headerSize + sizeof(float)*domainSize,SEEK_SET);
  fread(m_temperature,sizeof(float),domainSize,inFile);

  fclose(inFile);
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

int DensityTemperatureVolume::getRawDensityAt(int x, int y, int z)
{
  return getRawValueAt(m_density,x,y,z);
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
  int floorX=int(x), floorY=int(y), floorZ=int(z);
  int ceilX=int(x+1.f), ceilY=int(y+1.f), ceilZ=int(z+1.f);
  float xRatio=x-floor(x), yRatio=y-floor(y), zRatio=z-floor(z);
  float oneMinusXRatio=1.f-xRatio, oneMinusYRatio=1.f-yRatio, oneMinusZRatio=1.f-zRatio;
  return getRawValueAt(field,floorX,floorY,floorZ)*(oneMinusXRatio)*(oneMinusYRatio)*(oneMinusZRatio)+
         getRawValueAt(field,ceilX,floorY,floorZ)*(xRatio)*(oneMinusYRatio)*(oneMinusZRatio)+
         getRawValueAt(field,floorX,ceilY,floorZ)*(oneMinusXRatio)*(yRatio)*(oneMinusZRatio)+
         getRawValueAt(field,floorX,floorY,ceilZ)*(oneMinusXRatio)*(oneMinusYRatio)*(zRatio)+
         getRawValueAt(field,ceilX,floorY,ceilZ)*(xRatio)*(oneMinusYRatio)*(zRatio)+
         getRawValueAt(field,floorX,ceilY,ceilZ)*(oneMinusXRatio)*(yRatio)*(zRatio)+
         getRawValueAt(field,ceilX,ceilY,floorZ)*(xRatio)*(yRatio)*(oneMinusZRatio)+
         getRawValueAt(field,ceilX,ceilY,ceilZ)*(xRatio)*(yRatio)*(zRatio);
}

float DensityTemperatureVolume::getRawValueAt(const float* field, int x, int y, int z)
{
  return field[m_dimY*m_dimX*z + y*m_dimX + x];
}