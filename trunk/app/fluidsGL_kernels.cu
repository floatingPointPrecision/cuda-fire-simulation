/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include "fluidsGL_kernels.h"

// Texture reference for reading velocity field
texture<float2, 2> texref;
static cudaArray *array = NULL;

void setupTexture(int x, int y) {
    // Wrap mode appears to be the new default
    texref.filterMode = cudaFilterModeLinear;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

    cudaMallocArray(&array, &desc, y, x);
    cutilCheckMsg("cudaMalloc failed");
}

void bindTexture(void) {
    cudaBindTextureToArray(texref, array);
    cutilCheckMsg("cudaBindTexture failed");
}

void unbindTexture(void) {
    cudaUnbindTexture(texref);
    cutilCheckMsg("cudaUnbindTexture failed");
}
    
void updateTexture(cData *data, size_t wib, size_t h, size_t pitch) {
    cudaMemcpy2DToArray(array, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice);
    cutilCheckMsg("cudaMemcpy failed"); 
}

void deleteTexture(void) {
    cudaFreeArray(array);
    cutilCheckMsg("cudaFreeArray failed");
}

// CUDA FIRE KERNELS:

__global__ void addVelocityContribution(float2* vField, float2* newVField, int dim, int num_particles, bool replace)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index > num_particles)
    return;
  float2 velocity = newVField[index];
  if (!replace)
    velocity += vField[index];
  else
    velocity += (vField[index]*0.001f);
  vField[index] = velocity;
}

__global__ void clampVelocities(float2* vField, float maxLength, int numElements)
{
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= numElements)
    return;
  float2 velocity = vField[index];
  float length = sqrtf(velocity.x*velocity.x + velocity.y*velocity.y);
  if (length != 0)
  {
    velocity /= length;
    velocity *= min(maxLength, length);
  }
  vField[index] = velocity;
}

__global__ void dissipateDenistyKernel(float* densityField, float dissipationFactor, float dt, int fieldSize)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= fieldSize)
    return;
  float density = densityField[index];
  density = density * powf(1.f - dissipationFactor, dt);
  densityField[index] = max(0.f, density);
}

__global__ void dissipateFuelKernel(float* fuelField, float fuelDissipationFactor,float dt,int fieldSize)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= fieldSize)
    return;
  float fuel = fuelField[index];
  fuel = fuel * powf(1.f - fuelDissipationFactor, dt);
  fuelField[index] = fuel;
}

__global__ void coolTemperatureKernel(float* temperatureField,float coolingCoefficient,float maxTemperature,float dt,int fieldSize)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= fieldSize)
    return;
  float temperature = temperatureField[index];
  temperature = temperature - dt * coolingCoefficient * powf(temperature / maxTemperature, 4);
  temperatureField[index] = temperature;
}

__global__ void addMassFromSlice(float* densityField, float* mass, float densityFactor, int fieldSize)
{
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= fieldSize)
    return;
  float density = densityField[index];
  float massContribution = min(mass[index],1.f);
  density = max(density, massContribution * densityFactor);
  densityField[index] = density;
}

__global__ void addFuelFromSlice(float* fuelField,float* fuel,float combustionTemperature, int fieldSize)
{
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= fieldSize)
    return;
  float temperature = fuelField[index];
  float fuelContribution = min(fuel[index],1.f);
  temperature = max(temperature, combustionTemperature * fuelContribution);
  fuelField[index] = temperature;
}

__global__ void addTemperatureFromFuel(float* temperatureField,float* fuelField,float combustionTemperature,int numElements)
{
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= numElements)
    return;
  float temp = temperatureField[index];
  float fuel = min(fuelField[index], 1.f);
  temperatureField[index] = 1400.f;//max(temp, fuel*combustionTemperature);
}

// inclusive clamp
__device__ int clampToRange(int input, int lowerBound, int upperBound)
{
  if (input < lowerBound)
    input = lowerBound;
  else if (input >= upperBound)
    input = upperBound-1;
  return input;
}

__device__ int wrapInRange(int input, int lowerBound, int upperBound)
{
  int range = upperBound-lowerBound;
  return lowerBound + (input % range);
}

__device__ float bilinearInterpolation(float* field, float2 position, int dim)
{
  int x1 = floor(position.x); int x2 = ceil(position.x);
  int y1 = floor(position.y); int y2 = ceil(position.y);
  // periodic boundary conditions
  x1=clampToRange(x1,0,dim); x2=clampToRange(x2,0,dim);
  y1=clampToRange(y1,0,dim); y2=clampToRange(y2,0,dim);
  float q11 = field[y1*dim+x1]; float q12 = field[y1*dim+x2];
  float q21 = field[y2*dim+x1]; float q22 = field[y2*dim+x2];
  // taken from wikipedia http://en.wikipedia.org/wiki/Bilinear_interpolation
  float result = q11*(x2-position.x)*(y2-position.y) +
                 q21*(position.x-x1)*(y2-position.y) +
                 q12*(x2-position.x)*(position.y-y1) + 
                 q22*(position.x-x1)*(position.y-y1);
  return result;
          
}

__global__ void semiLagrangianAdvectionKernel(float2* velocityField, float* scalarField, float* tempScalarField, float dt, int numElements, int dim)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElements)
    return;
  float2 velocity = velocityField[index];
  int col = index % dim;
  int row = index / dim;
  float2 newPos = make_float2(col,row) - velocity * dt;
  tempScalarField[index] = bilinearInterpolation(scalarField, newPos, dim);
}

__device__ float2 partialDerivativeYFloat2(float2* vField, int2 middleIndex, int dim)
{
  int2 upIndex = middleIndex + make_int2(0,1);
  upIndex.y = upIndex.y % dim;
  float2 up = vField[upIndex.y*dim+upIndex.x];
  int2 downIndex = middleIndex + make_int2(0,-1);
  if (downIndex.y < 0)
    downIndex.y += dim;
  float2 down = vField[downIndex.y*dim+downIndex.x];
  return (up - down) / 2.f;
}

__device__ float2 partialDerivativeXFloat2(float2* vField, int2 middleIndex, int dim)
{
  int2 rightIndex = middleIndex + make_int2(1,0);
  rightIndex.x = rightIndex.x % dim;
  float2 right = vField[rightIndex.y*dim+rightIndex.x];
  int2 leftIndex = middleIndex + make_int2(-1,0);
  if (leftIndex.y < 0)
    leftIndex.y += dim;
  float2 left = vField[leftIndex.y*dim+leftIndex.x];
  return (right - left) / 2.f;
}

__global__ void calculatePhi(float2* vField, float* phi, float* turbulenceField, float vorticityTerm, int numElements, int dim)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElements)
    return;
  int row = index / dim;
  int col = index % dim;
  float2 Vx = partialDerivativeXFloat2(vField, make_int2(col,row), dim);
  float2 Vy = partialDerivativeYFloat2(vField, make_int2(col,row), dim);
  float psi = Vy.x - Vx.y;
  phi[index] = turbulenceField[index] + (psi * vorticityTerm);
}

__device__ float partialDerivativeYFloat(float* field, int2 middleIndex, int dim)
{
  int2 upIndex = middleIndex + make_int2(0,1);
  upIndex.y = upIndex.y % dim;
  float up = field[upIndex.y*dim+upIndex.x];
  int2 downIndex = middleIndex + make_int2(0,-1);
  if (downIndex.y < 0)
    downIndex.y += dim;
  float down = field[downIndex.y*dim+downIndex.x];
  return (up - down) / 2.f;
}

__device__ float partialDerivativeXFloat(float* field, int2 middleIndex, int dim)
{
  int2 rightIndex = middleIndex + make_int2(1,0);
  rightIndex.x = rightIndex.x % dim;
  float right = field[rightIndex.y*dim+rightIndex.x];
  int2 leftIndex = middleIndex + make_int2(-1,0);
  if (leftIndex.y < 0)
    leftIndex.y += dim;
  float left = field[leftIndex.y*dim+leftIndex.x];
  return (right - left) / 2.f;
}

__global__ void vorticityConfinementTurbulence(float2* vField,float* phi,float dt,int numElements,int dim)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numElements)
    return;
  int row = index / dim;
  int col = index % dim;
  float2 velocityTerm = make_float2(partialDerivativeYFloat(phi,make_int2(col,row),dim),
    -partialDerivativeXFloat(phi,make_int2(col,row),dim));
  vField[index] += dt * velocityTerm;
}

// OLD FFT SOLVER KERNELS:

// Note that these kernels are designed to work with arbitrary 
// domain sizes, not just domains that are multiples of the tile
// size. Therefore, we have extra code that checks to make sure
// a given thread location falls within the domain boundaries in
// both X and Y. Also, the domain is covered by looping over
// multiple elements in the Y direction, while there is a one-to-one
// mapping between threads in X and the tile size in X.
// Nolan Goodnight 9/22/06

// This method adds constant force vectors to the velocity field 
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void 
addForces_k(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    cData *fj = (cData*)((char*)v + (ty + spy) * pitch) + tx + spx;

    cData vterm = *fj;
    tx -= r; ty -= r;
    float s = 1.f / (1.f + tx*tx*tx*tx + ty*ty*ty*ty);
    vterm.x += s * fx;
    vterm.y += s * fy;
    *fj = vterm;
}

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void 
advectVelocity_k(cData *v, float *vx, float *vy,
                 int dx, int pdx, int dy, float dt, int lb) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    cData vterm, ploc;
    float vxterm, vyterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * pdx + gtidx;
                vterm = tex2D(texref, (float)gtidx, (float)fi);
                ploc.x = (gtidx + 0.5f) - (dt * vterm.x * dx);
                ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
                vterm = tex2D(texref, ploc.x, ploc.y);
                vxterm = vterm.x; vyterm = vterm.y; 
                vx[fj] = vxterm;
                vy[fj] = vyterm; 
            }
        }
    }
}

// This method performs velocity diffusion and forces mass conservation 
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued 
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void 
diffuseProject_k(cData *vx, cData *vy, int dx, int dy, float dt, 
                 float visc, int lb) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    cData xterm, yterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * dx + gtidx;
                xterm = vx[fj];
                yterm = vy[fj];

                // Compute the index of the wavenumber based on the
                // data order produced by a standard NN FFT.
                int iix = gtidx;
                int iiy = (fi>dy/2)?(fi-(dy)):fi;

                // Velocity diffusion
                float kk = (float)(iix * iix + iiy * iiy); // k^2 
                float diff = 1.f / (1.f + visc * dt * kk);
                xterm.x *= diff; xterm.y *= diff;
                yterm.x *= diff; yterm.y *= diff;

                // Velocity projection
                if (kk > 0.f) {
                    float rkk = 1.f / kk;
                    // Real portion of velocity projection
                    float rkp = (iix * xterm.x + iiy * yterm.x);
                    // Imaginary portion of velocity projection
                    float ikp = (iix * xterm.y + iiy * yterm.y);
                    xterm.x -= rkk * rkp * iix;
                    xterm.y -= rkk * ikp * iix;
                    yterm.x -= rkk * rkp * iiy;
                    yterm.y -= rkk * ikp * iiy;
                }
                
                vx[fj] = xterm;
                vy[fj] = yterm;
            }
        }
    }
}

// This method updates the velocity field 'v' using the two complex 
// arrays from the previous step: 'vx' and 'vy'. Here we scale the 
// real components by 1/(dx*dy) to account for an unnormalized FFT. 
__global__ void 
updateVelocity_k(cData *v, float *vx, float *vy, 
                 int dx, int pdx, int dy, int lb, size_t pitch) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    float vxterm, vyterm;
    cData nvterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fjr = fi * pdx + gtidx; 
                vxterm = vx[fjr];
                vyterm = vy[fjr];

                // Normalize the result of the inverse FFT
                float scale = 1.f / (dx * dy);
                nvterm.x = vxterm * scale;
                nvterm.y = vyterm * scale;
               
                cData *fj = (cData*)((char*)v + fi * pitch) + gtidx;
                *fj = nvterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).  
__global__ void 
advectParticles_k(cData *part, cData *v, int dx, int dy, 
                  float dt, int lb, size_t pitch) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    // gtidx is the domain location in x for this thread
    cData pterm, vterm;
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * dx + gtidx;
                pterm = part[fj];
                
                int xvi = ((int)(pterm.x * dx));
                int yvi = ((int)(pterm.y * dy));
                vterm = *((cData*)((char*)v + yvi * pitch) + xvi);   
 
                pterm.x += dt * vterm.x;
                pterm.x = pterm.x - (int)pterm.x;            
                pterm.x += 1.f; 
                pterm.x = pterm.x - (int)pterm.x;              
                pterm.y += dt * vterm.y;
                pterm.y = pterm.y - (int)pterm.y;            
                pterm.y += 1.f; 
                pterm.y = pterm.y - (int)pterm.y;                  

                part[fj] = pterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}
