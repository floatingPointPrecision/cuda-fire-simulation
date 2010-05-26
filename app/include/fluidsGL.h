#pragma once

#include <vector_types.h>

void setupSliceVisualization(int argc, char** argv);
void setupSliceSimulation();
void sliceDisplay();
void simulateFluids(float dt);

void replaceVelocityField(float2* velocityField);
void dissipateDensity(float dt);
void dissipateFuel(float dt);
void coolTemperature(float dt);
void contributeSlices(float* mass, float* fuel);

float* getDensityField();
float* getTemperatureField();
float* getTextureField();
float* getFuelField();