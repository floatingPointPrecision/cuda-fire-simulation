#pragma once

#include <vector_types.h>

enum Slice
{
  SliceTexture = 0,
  SliceDensity,
  SliceFuel,
  SliceTemperature,
  SliceVelocity
};

void setupSliceVisualization(int argc, char** argv);
void setupSliceSimulation();
void displaySlice(int slice);
void sliceDisplay();
void simulateFluids(float dt);
void enforveVelocityIncompressibility(float dt);

void addVelocity(float2* velocityField);
void dissipateDensity(float dt);
void dissipateFuel(float dt);
void coolTemperature(float dt);
void contributeSlices(float* mass, float* fuel);
void addTextureDetail(float time, float zVal);
void addTurbulenceVorticityConfinement(float time, float zVal, float dt);

float* getDensityField();
float* getTemperatureField();
float* getTextureField();
float* getFuelField();