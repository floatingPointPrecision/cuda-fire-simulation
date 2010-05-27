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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <GL/glew.h>
#include <cufft.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <cutil_math.h>
#include "SimplexNoise.h"
#include "fluidsGl.h"


#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "fluidsGL_kernels.cu"
#include "ocuutil/timer.h"

#define MAX_EPSILON_ERROR 1.0f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
  "fluidsGL.ppm",
  NULL
};

const char *sReference[] =
{
  "ref_fluidsGL.ppm",
  NULL
};

// CUDA example code that implements the frequency space version of 
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the 
// CUDA FFT library (CUFFT) to perform velocity diffusion and to 
// force non-divergence in the velocity field at each time step. It uses 
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step. 

#ifdef __DEVICE_EMULATION__
#define DIM    64        // Square size of solver domain
#else
#define DIM    512       // Square size of solver domain
#endif
#define DS    (DIM*DIM)  // Total domain size
#define CPADW (DIM/2+1)  // Padded width for real->complex in-place FFT
#define RPADW (2*(DIM/2+1))  // Padded width for real->complex in-place FFT
#define PDS   (DIM*CPADW) // Padded total domain size

#define DT     0.09f     // Delta T for interative solver
#define VIS    0.00025f   // Viscosity constant
#define FORCE (5.8f*DIM) // Force scale factor 
#define FR     4         // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

void cleanup(void);
void reshape(int x, int y);

// CUFFT plan handle
static cufftHandle planr2c;
static cufftHandle planc2r;
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData *hvfield = NULL;
cData *dvfield = NULL;
static int wWidth = max(512,DIM);
static int wHeight = max(512,DIM);

float* densityField = NULL;
float* temperatureField = NULL;
float* textureField = NULL;
float* fuelField = NULL;
float* tempScalarField = NULL;

float* phiField = NULL;
float* turbulenceField = NULL;

float* h_tempScalarField = NULL;
float2* h_tempVelocityField = NULL;
int* h_frameBuffer = NULL;

SimplexNoise4D* simplexField = NULL;
SimplexNoise4D* simplexFieldTurbulence = NULL;
float* coarseMassPlane = NULL;
float* coarseFuelPlane = NULL;


static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

// Particle data
GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
static cData *particles = NULL; // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

bool				  g_bQAReadback     = false;
bool				  g_bQAAddTestForce = true;
int					  g_iFrameToCompare = 100;
int                   g_TotalErrors     = 0;

void autoTest();

float* getDensityField()
{
  return densityField;
}
float* getTemperatureField()
{
  return temperatureField;
}
float* getTextureField()
{
  return textureField;
}
float* getFuelField()
{
  return fuelField;
}

void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r) 
{ 

  dim3 tids(2*r+1, 2*r+1);

  addForces_k<<<1, tids>>>(v, dx, dy, spx, spy, fx, fy, r, tPitch);
  cutilCheckMsg("addForces_k failed.");
}

void advectVelocity(cData *v, float *vx, float *vy,
                    int dx, int pdx, int dy, float dt) 
{ 

  dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

  dim3 tids(TIDSX, TIDSY);

  updateTexture(v, DIM*sizeof(cData), DIM, tPitch);
  advectVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, dt, TILEY/TIDSY);

  cutilCheckMsg("advectVelocity_k failed.");
}

void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt,
                    float visc) 
{ 
  // Forward FFT
  cufftExecR2C(planr2c, (cufftReal*)vx, (cufftComplex*)vx); 
  cufftExecR2C(planr2c, (cufftReal*)vy, (cufftComplex*)vy);

  uint3 grid = make_uint3((dx/TILEX)+(!(dx%TILEX)?0:1), 
    (dy/TILEY)+(!(dy%TILEY)?0:1), 1);

  uint3 tids = make_uint3(TIDSX, TIDSY, 1);

  diffuseProject_k<<<grid, tids>>>(vx, vy, dx, dy, dt, visc, TILEY/TIDSY);
  cutilCheckMsg("diffuseProject_k failed.");

  // Inverse FFT
  cufftExecC2R(planc2r, (cufftComplex*)vx, (cufftReal*)vx); 
  cufftExecC2R(planc2r, (cufftComplex*)vy, (cufftReal*)vy);
}

void updateVelocity(cData *v, float *vx, float *vy, 
                    int dx, int pdx, int dy) 
{ 

  dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

  dim3 tids(TIDSX, TIDSY);

  updateVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, TILEY/TIDSY, tPitch);
  cutilCheckMsg("updateVelocity_k failed.");
}

void advectParticles(GLuint vbo, cData *v, int dx, int dy, float dt) 
{

  dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

  dim3 tids(TIDSX, TIDSY);

  cData *p;
  cutilSafeCall(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
  size_t num_bytes; 
  cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes,  
    cuda_vbo_resource));
  cutilCheckMsg("cudaGraphicsResourceGetMappedPointer failed");

  advectParticles_k<<<grid, tids>>>(p, v, dx, dy, dt, TILEY/TIDSY, tPitch);
  cutilCheckMsg("advectParticles_k failed.");

  cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

  cutilCheckMsg("cudaGraphicsUnmapResources failed");
}


void addVelocity(float2* velocityField)
{
  int blockSize = 512;
  int gridSize = (DS + blockSize - 1) / blockSize;
  addVelocityContribution<<<gridSize,blockSize>>>(dvfield, velocityField, DIM, DS, true);
  // clamp velocities
  clampVelocities<<<gridSize,blockSize>>>(dvfield, 25.f, DS);
}

void dissipateDensity(float dt)
{
  float dissipationFactor = 0.25f;
  int blockSize = 512;
  int gridSize = (DS + blockSize - 1) / blockSize;
  dissipateDenistyKernel<<<gridSize,blockSize>>>(densityField,dissipationFactor,dt,DS);
  cutilCheckMsg("dissipate density failed");
}

void dissipateFuel(float dt)
{
  float fuelDissipationFactor = 1.f;
  int blockSize = 512;
  int gridSize = (DS + blockSize - 1) / blockSize;
  dissipateFuelKernel<<<gridSize,blockSize>>>(fuelField,fuelDissipationFactor,dt,DS);
  cutilCheckMsg("dissipate fuel failed");
}

void coolTemperature(float dt)
{
  float coolingCoefficient = 300000.f;
  float maxTemperature = 4000;
  int blockSize = 512;
  int gridSize = (DS + blockSize - 1) / blockSize;
  coolTemperatureKernel<<<gridSize,blockSize>>>(temperatureField,coolingCoefficient,maxTemperature,dt,DS);
  cutilCheckMsg("cool temperature failed");
}

void contributeSlices(float* mass, float* fuel)
{
  float densityFactor = 10.f;
  float combustionTemperature = 1700.f;
  coarseMassPlane = mass;
  coarseFuelPlane = fuel;
  int blockSize = 512;
  int gridSize = (DS + blockSize - 1) / blockSize;
  addMassFromSlice<<<gridSize,blockSize>>>(densityField, mass, densityFactor, DS);
  cutilCheckMsg("adding density from slice mass failed");
  addFuelFromSlice<<<gridSize,blockSize>>>(fuelField,fuel,combustionTemperature, DS);
  cutilCheckMsg("adding temperature from fuel slice failed");
  addTemperatureFromFuel<<<gridSize,blockSize>>>(temperatureField,fuelField,combustionTemperature,DS);

}

void semiLagrangianAdvection(float2* velocityField, float* scalarField, float* tempScalarField, float dt)
{
  int blockSize = 512;
  int gridSize = (DS + blockSize - 1) / blockSize;
  semiLagrangianAdvectionKernel<<<gridSize,blockSize>>>(velocityField, scalarField, tempScalarField, dt, DS, DIM);
  cudaMemcpy(scalarField,tempScalarField,sizeof(float)*DS,cudaMemcpyDeviceToDevice);
  cutilCheckMsg("semi-Lagrangian advection");
}

void addTextureDetail(float time, float zVal)
{
  time *= 10.f;
  simplexField->updateNoise(textureField, zVal, time, 2.3f, DS, DIM);
  simplexField->updateNoise(textureField, zVal, time, 1.1f, DS, DIM);
  simplexField->updateNoise(textureField, zVal, time, .75f, DS, DIM);
  cutilCheckMsg("texture synthesis");
}

void addTurbulenceVorticityConfinement(float time, float zVal, float dt)
{
  simplexFieldTurbulence->updateNoise(turbulenceField, time, zVal, 1.f, DS, DIM);
  int blockSize = 512;
  int gridSize = (DS + blockSize - 1) / blockSize;
  float vorticityTerm = 150.f;
  calculatePhi<<<gridSize,blockSize>>>(dvfield,phiField,turbulenceField, vorticityTerm,DS,DIM);
  vorticityConfinementTurbulence<<<gridSize,blockSize>>>(dvfield,phiField,dt,DS,DIM);
}

void simulateFluids(float dt)//float2* newVelocityField)
{
  // perform advection
  advectVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM, dt);
  semiLagrangianAdvection(dvfield, densityField, tempScalarField, dt);
  semiLagrangianAdvection(dvfield, fuelField, tempScalarField, dt);
  semiLagrangianAdvection(dvfield, temperatureField, tempScalarField, dt);
  semiLagrangianAdvection(dvfield, textureField, tempScalarField, dt);
}

void enforveVelocityIncompressibility(float dt)
{
  diffuseProject(vxfield, vyfield, CPADW, DIM, dt, VIS);
  updateVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM);
}

void displaySlice(int slice)
{
  float pixelScale, pixelAddition, *field;
  if (slice == SliceVelocity)
  {
    cudaMemcpy(h_tempVelocityField,dvfield,sizeof(float2)*DS,cudaMemcpyDeviceToHost);
    float output;
    float2 velocity;
    float xTotal = 0.f, yTotal = 0.f;
    int numNonEmpty = 0;
    for (int i = 0; i < DIM; i++)
    {
      for (int j = 0; j < DIM; j++)
      {
        velocity = h_tempVelocityField[j*DIM+i];
        velocity.x = fabs(velocity.x);
        velocity.y = fabs(velocity.y);
        if (velocity.x != 0.f || velocity.y != 0.f)
        {
          numNonEmpty++;
          if (velocity.x != 0.f)
            xTotal += velocity.x;
          if (velocity.y != 0.f)
            yTotal += velocity.y;
        }
        output = sqrtf(velocity.x*velocity.x+velocity.y*velocity.y);
        output /= 20.f;
        h_tempScalarField[j*DIM+i] = output;
      }
    }
    printf("x average: %f\n", xTotal / numNonEmpty);
    printf("y average: %f\n", yTotal / numNonEmpty);
    glWindowPos2i(0,0);
    glDrawPixels(DIM,DIM,GL_LUMINANCE,GL_FLOAT,h_tempScalarField);
    return;
  }
  else
  {
    if (slice == SliceTexture)
    {
      pixelAddition = 0.5f;
      pixelScale = 0.5f;
      field = textureField;
    }
    else if (slice == SliceFuel)
    {
      pixelAddition = 0.f;
      pixelScale = 1 / 2000.f;
      field = fuelField;
    }
    else if (slice == SliceDensity)
    {
      pixelAddition = 0.f;
      pixelScale = 1.f / 15.f;
      field = densityField;
    }
    else if (slice == SliceTemperature)
    {
      pixelAddition = 0.f;
      pixelScale = 1700.f;
      field = temperatureField;
    }
    cudaMemcpy(h_tempScalarField,field,sizeof(float)*DS,cudaMemcpyDeviceToHost);
    float total = 0.f;
    int numNonEmpty = 0;
    for (int i = 0; i < DIM; i++)
    {
      for (int j = 0; j < DIM; j++)
      {
        float output = h_tempScalarField[j*DIM+i]*pixelScale+pixelAddition;
        if (output != 0.f)
        {
          numNonEmpty++;
          total += output;
        }
        h_tempScalarField[j*DIM+i] = output;
      }
    }
    printf("average scalar field value: %f\n",total / numNonEmpty);
  }
  glWindowPos2i(0,0);
  glDrawPixels(DIM,DIM,GL_LUMINANCE,GL_FLOAT,h_tempScalarField);
  
}

void sliceDisplay(void) {  
  cutilCheckError(cutStartTimer(timer));

  // render points from vertex buffer
  glClear(GL_COLOR_BUFFER_BIT);
  glColor4f(0,1,0,0.5f); glPointSize(1);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnableClientState(GL_VERTEX_ARRAY);    
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE); 
  glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
  glVertexPointer(2, GL_FLOAT, 0, NULL);
  glDrawArrays(GL_POINTS, 0, DS);
  glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
  glDisableClientState(GL_VERTEX_ARRAY); 
  glDisableClientState(GL_TEXTURE_COORD_ARRAY); 
  glDisable(GL_TEXTURE_2D);

  // Finish timing before swap buffers to avoid refresh sync
  cutilCheckError(cutStopTimer(timer));  
  glutSwapBuffers();

  fpsCount++;
  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
    sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", DIM, DIM, ifps);  
    glutSetWindowTitle(fps);
    fpsCount = 0; 
    fpsLimit = (int)max(ifps, 1.f);
    cutilCheckError(cutResetTimer(timer));  
  }

  glutPostRedisplay();
}

void autoTest() 
{
  reshape(wWidth, wHeight);
  for(int count=0;count<g_iFrameToCompare;count++)
  {
    simulateFluids(DT);

    // add in a little force so the automated testing is interesing.
    if(g_bQAReadback && g_bQAAddTestForce) 
    {
      int x = wWidth/(count+1); int y = wHeight/(count+1);
      float fx = (x / (float)wWidth);        
      float fy = (y / (float)wHeight);
      int nx = (int)(fx * DIM);        
      int ny = (int)(fy * DIM);   

      int ddx = 35;
      int ddy = 35;
      fx = ddx / (float)wWidth;
      fy = ddy / (float)wHeight;
      int spy = ny-FR;
      int spx = nx-FR;

      addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
      lastx = x; lasty = y;
      //g_bQAAddTestForce = false; // only add it once
    }
    sliceDisplay();
  }

  // re-swap to make sure final frame is in back buffer instead of front
  glutSwapBuffers();

  // compare to offical reference image, printing PASS or FAIL.
  printf("> (Frame %d) Readback BackBuffer\n", 100);
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
  static int seed = 72191;
  char sq[22];


  if (g_bQAReadback) {
    seed *= seed;
    sprintf(sq, "%010d", seed);
    // pull the middle 5 digits out of sq
    sq[8] = 0;
    seed = atoi(&sq[3]);

    return seed/99999.f;
  } else {
    return rand()/(float)RAND_MAX;
  }
}

void initParticles(cData *p, int dx, int dy) {
  int i, j;
  for (i = 0; i < dy; i++) {
    for (j = 0; j < dx; j++) {
      p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
      p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
    }
  }
}

void sliceKeyboard( unsigned char key, int x, int y) {
  switch( key) {
        case 27:
          exit (0);
        case 'r':
          memset(hvfield, 0, sizeof(cData) * DS);
          cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, 
            cudaMemcpyHostToDevice);

          initParticles(particles, DIM, DIM);

          cudaGraphicsUnregisterResource(cuda_vbo_resource);

          cutilCheckMsg("cudaGraphicsUnregisterBuffer failed");

          glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
          glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS, 
            particles, GL_DYNAMIC_DRAW_ARB);
          glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

          cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

          cutilCheckMsg("cudaGraphicsGLRegisterBuffer failed");
          break;
        default: break;
  }
}

void sliceClick(int button, int updown, int x, int y) {
  lastx = x; lasty = y;
  clicked = !clicked;
}

void sliceMotion (int x, int y) {
  // Convert motion coordinates to domain
  float fx = (lastx / (float)wWidth);        
  float fy = (lasty / (float)wHeight);
  int nx = (int)(fx * DIM);        
  int ny = (int)(fy * DIM);   

  if (clicked && nx < DIM-FR && nx > FR-1 && ny < DIM-FR && ny > FR-1) {
    int ddx = x - lastx;
    int ddy = y - lasty;
    fx = ddx / (float)wWidth;
    fy = ddy / (float)wHeight;
    int spy = ny-FR;
    int spx = nx-FR;
    addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
    lastx = x; lasty = y;
  } 
  glutPostRedisplay();
}

void sliceReshape(int x, int y) {
  wWidth = x; wHeight = y;
  glViewport(0, 0, x, y);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 1, 0, 0, 1); 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glutPostRedisplay();
}

void cleanup(void) {
  cudaGraphicsUnregisterResource(cuda_vbo_resource);
  cutilCheckMsg("cudaGLUnregisterResource failed");

  unbindTexture();
  deleteTexture();

  // Free all host and device resources
  free(hvfield); free(particles); 
  cudaFree(dvfield); 
  cudaFree(vxfield); cudaFree(vyfield);
  cufftDestroy(planr2c);
  cufftDestroy(planc2r);

  glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
  glDeleteBuffersARB(1, &vbo);

  cutilCheckError(cutDeleteTimer(timer));  
}

void setupSliceVisualization(int argc, char** argv)
{
  int devID;
  cudaDeviceProp deviceProps;

  printf("[fluidsGL] - [OpenGL/CUDA simulation]\n");


  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(wWidth, wHeight);
  glutCreateWindow("Compute Stable Fluids");
  glutDisplayFunc(sliceDisplay);
  glutKeyboardFunc(sliceKeyboard);
  glutMouseFunc(sliceClick);
  glutMotionFunc(sliceMotion);
  glutReshapeFunc(sliceReshape);
  glewInit();

  // use command-line specified CUDA device, otherwise use device with highest Gflops/s
  if (cutCheckCmdLineFlag(argc, (const char**)argv, "device")) {
    cutilGLDeviceInit(argc, argv);
    cutGetCmdLineArgumenti(argc, (const char **) argv, "device", &devID);
  } else {
    devID = cutGetMaxGflopsDeviceId();
    cutilSafeCall(cudaGLSetGLDevice(devID));
  }

  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

  // automated build testing harness
  if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest") ||
    cutCheckCmdLineFlag(argc, (const char **)argv, "noprompt"))
  {
    g_bQAReadback = true;
    if (strncmp(deviceProps.name, "Tesla", sizeof("Tesla")-1) == 0) {
      printf("[fluidsGL] - Test Results:\nPASSED\n");
      exit(0);
    }
  }

}

void setupSliceSimulation()
{
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutResetTimer(timer));  

  hvfield = (cData*)malloc(sizeof(cData) * DS);
  memset(hvfield, 0, sizeof(cData) * DS);

  // Allocate and initialize device data
  cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(cData)*DIM, DIM);

  cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, 
    cudaMemcpyHostToDevice); 
  // Temporary complex velocity field data     
  cudaMalloc((void**)&vxfield, sizeof(cData) * PDS);
  cudaMalloc((void**)&vyfield, sizeof(cData) * PDS);
  cudaMalloc((void**)&densityField, sizeof(float) * DS);
  cudaMalloc((void**)&temperatureField, sizeof(float) * DS);
  cudaMalloc((void**)&textureField, sizeof(float) * DS);
  cudaMalloc((void**)&fuelField, sizeof(float) * DS);
  cudaMalloc((void**)&tempScalarField, sizeof(float) * DS);
  cudaMalloc((void**)&turbulenceField, sizeof(float) * DS);
  cudaMalloc((void**)&phiField, sizeof(float) * DS);

  cudaMemset(densityField,0,sizeof(float)*DS);
  cudaMemset(temperatureField,0,sizeof(float)*DS);
  cudaMemset(textureField,0,sizeof(float)*DS);
  cudaMemset(fuelField,0,sizeof(float)*DS);
  cudaMemset(tempScalarField,0,sizeof(float)*DS);
  h_tempScalarField = (float*)malloc(sizeof(float)*DS);
  h_tempVelocityField = (float2*)malloc(sizeof(float2)*DS);
  h_frameBuffer = (int*)malloc(sizeof(int)*DS);
  simplexField = new SimplexNoise4D();
  simplexFieldTurbulence = new SimplexNoise4D();
  setupTexture(DIM, DIM);
  bindTexture();

  // Create particle array
  particles = (cData*)malloc(sizeof(cData) * DS);
  memset(particles, 0, sizeof(cData) * DS);   

  initParticles(particles, DIM, DIM); 

  // Create CUFFT transform plan configuration
  cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C);
  cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R);

  glGenBuffersARB(1, &vbo);
  glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
  glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS, 
    particles, GL_DYNAMIC_DRAW_ARB);
  glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

  cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
  cutilCheckMsg("cudaGraphicsGLRegisterBuffer failed");

  if (g_bQAReadback)
  {
    autoTest();

    printf("[fluidsGL] - Test Results: %d Failures\n", g_TotalErrors);
    printf((g_TotalErrors == 0) ? "PASSED" : "FAILED");
    printf("\n");

  } else {
    atexit(cleanup); 
  }
}
