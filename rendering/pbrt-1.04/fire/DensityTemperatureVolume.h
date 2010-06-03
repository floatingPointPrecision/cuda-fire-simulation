#pragma once

class DensityTemperatureVolume
{
public:
  DensityTemperatureVolume(const char* fileName);
  virtual ~DensityTemperatureVolume();

  virtual float getDensityAt(float x, float y, float z);
  virtual float getTempAt(float x, float y, float z);
  
  int getSizeInX();
  int getSizeInY();
  int getSizeInZ();

protected:
  float trilinearlyInterpolate(const float* field, float x, float y, float z);
  float getRawValueAt(const float* field, int x, int y, int z);

protected:
  int m_dimX;
  int m_dimY;
  int m_dimZ;
  float* m_density;
  float* m_temperature;
};