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
@file CoarseParticleVisualization.cu
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <math.h>
//#include <iostream>
//#include <sstream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glu.h>

// includes, CUDA
#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_math.h>
#include "ocuutil/timer.h"

// includes, CUFIRE
#include "XMLParser.h"
#include "CoarseParticleEngine.h"
#include "Projection.h"
#include "3DNavierStokes.h"
#include "Bitmap.h"
#include "SliceManager.h"
#include "Renderer.h"

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;

using namespace cufire;

int enableCoarseVisualization;

int numRenderTargets = 8;
enum RenderTarget
{
  RenderTexture = 0,
  RenderDensity,
  RenderFuel,
  RenderTemperature,
  RenderVelocity,
  RenderCoarseEngine,
  RenderSingleSlice,
  RenderComposite
};
std::string currentRenderTargetString = "Slice Texture";

int currentRenderTarget = RenderTexture;
int currentSliceToDisplay;
CoarseParticleEngine* pEngine;
OrthographicProjection* pProjection;
SliceManager* pSliceManager;
Renderer* pRenderer;

float sTimestep;
float currentTime = 0;
bool pauseSimulation = false;

float* d_sliceMassOutput;
float* d_sliceFuelOutput;
float2* d_sliceVelocityOutput;

float* h_sliceMassOutput;
float* h_sliceFuelOutput;
float2* h_sliceVelocityOutput;
int numSliceBytes, numSliceVelocityBytes;
int2 slicePixelDims;
float imageSize;
float2 zBBox;
int numSlices;
//angle of rotation
float xpos = 32, ypos = 32, zpos = 90, xrot = 0, yrot = 0, angle=0.0;
float3 cameraTarget, cameraUp;
float theta, phi, cameraDistance;

//SliceRefinement* sliceRefinementTest;


// rendering callbacks
void display();
void reshape(int w, int h);

float randomNormalizedFloat()
{
  return (float(rand()) / RAND_MAX);
}

float randomFloatInRange(float minVal, float maxVal)
{
  float range = maxVal - minVal;
  float returnVal = minVal + randomNormalizedFloat() * range;
  if (rand()%2==0) 
    returnVal *= -1.0f;
  return returnVal;
}

void updateSimulation(float dt)
{
  currentTime += dt;
  printf("\n\n");
  // move VBO from OpenGL context to CUDA context
  pEngine->enableCUDAVbo();

  // update coarse simulation particles
  static float coarseSum = 0.f;
  CPUTimer coarseTimer;
  cudaThreadSynchronize();
  coarseTimer.start();
  pEngine->advanceSimulation(dt);
  cudaThreadSynchronize();
  coarseTimer.stop();
  coarseSum += coarseTimer.elapsed_sec();
  //printf("average coarse simulation time: %f\n",coarseSum / (currentTime/dt));

  // project coarse particles onto slices
  pProjection->setParticles(pEngine->getParticleBegins(), pEngine->getNumParticles());
  pSliceManager->startUpdateSeries();
  for (int i = 0; i < numSlices; i++)
  {
    float zIntercept = zBBox.x + ((zBBox.y-zBBox.x)/numSlices)*i;

    cutilSafeCall(cudaMemset(d_sliceMassOutput,0,numSliceBytes));
    cutilSafeCall(cudaMemset(d_sliceFuelOutput,0,numSliceBytes));
    cutilSafeCall(cudaMemset(d_sliceVelocityOutput,0,numSliceVelocityBytes));

    pProjection->execute(zIntercept, d_sliceMassOutput, d_sliceFuelOutput, d_sliceVelocityOutput);

    pSliceManager->updateIndividualSlice(i, d_sliceVelocityOutput, d_sliceMassOutput, d_sliceFuelOutput);
  }
  // move VBO back to OpenGL
  pEngine->disableCUDAVbo();
}

void drawCube(float3 lowerLeftFront, float3 upperRightBack)
{
  glBegin(GL_QUADS);
  // front face
  glColor4f(1,0,0,0.1f);
  glVertex3f(lowerLeftFront.x,lowerLeftFront.y,lowerLeftFront.z);
  glVertex3f(lowerLeftFront.x,upperRightBack.y,lowerLeftFront.z);
  glVertex3f(upperRightBack.x,upperRightBack.y,lowerLeftFront.z);
  glVertex3f(upperRightBack.x,lowerLeftFront.y,lowerLeftFront.z);
  // back face
  glVertex3f(lowerLeftFront.x,lowerLeftFront.y,upperRightBack.z);
  glVertex3f(lowerLeftFront.x,upperRightBack.y,upperRightBack.z);
  glVertex3f(upperRightBack.x,upperRightBack.y,upperRightBack.z);
  glVertex3f(upperRightBack.x,lowerLeftFront.y,upperRightBack.z);
  // left face
  glColor4f(0,1,0,0.1f);
  glVertex3f(lowerLeftFront.x,lowerLeftFront.y,lowerLeftFront.z);
  glVertex3f(lowerLeftFront.x,lowerLeftFront.y,upperRightBack.z);
  glVertex3f(lowerLeftFront.x,upperRightBack.y,upperRightBack.z);
  glVertex3f(lowerLeftFront.x,upperRightBack.y,lowerLeftFront.z);
  // right face
  glVertex3f(upperRightBack.x,lowerLeftFront.y,lowerLeftFront.z);
  glVertex3f(upperRightBack.x,lowerLeftFront.y,upperRightBack.z);
  glVertex3f(upperRightBack.x,upperRightBack.y,upperRightBack.z);
  glVertex3f(upperRightBack.x,upperRightBack.y,lowerLeftFront.z);
  // bottom face
  glColor4f(0,0,1,0.1f);
  glVertex3f(upperRightBack.x,upperRightBack.y,upperRightBack.z);
  glVertex3f(lowerLeftFront.x,upperRightBack.y,upperRightBack.z);
  glVertex3f(lowerLeftFront.x,upperRightBack.y,lowerLeftFront.z);
  glVertex3f(upperRightBack.x,upperRightBack.y,lowerLeftFront.z);
  // top face
  glVertex3f(upperRightBack.x,lowerLeftFront.y,upperRightBack.z);
  glVertex3f(lowerLeftFront.x,lowerLeftFront.y,upperRightBack.z);
  glVertex3f(lowerLeftFront.x,lowerLeftFront.y,lowerLeftFront.z);
  glVertex3f(upperRightBack.x,lowerLeftFront.y,lowerLeftFront.z);

  glEnd();
}

#define M_PI 3.14159265f
void display()
{
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  float radianTheta = theta * M_PI / 180.f;
  float radianPhi = phi * M_PI / 180.f;
  float x = cameraTarget.x + cameraDistance * sinf(radianTheta) * cosf(radianPhi);
  float y = cameraTarget.y + cameraDistance * sinf(radianTheta) * sinf(radianPhi);
  float z = cameraTarget.z + cameraDistance * cosf(radianTheta);
  gluLookAt(x,y,z, 
    cameraTarget.x,cameraTarget.y,cameraTarget.z, 
    cameraUp.x,cameraUp.y,cameraUp.z);

  // draw bounding box
  drawCube(make_float3(0,0,0), make_float3(64,64,64));
  pEngine->render();
}

// found at http://www-course.cs.york.ac.uk/cgv/OpenGL/L23b.html
void DrawText(GLint x, GLint y, char* s, GLfloat r, GLfloat g, GLfloat b)
{
  int lines;
  char* p;
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, glutGet(GLUT_WINDOW_WIDTH), 
    0.0, glutGet(GLUT_WINDOW_HEIGHT), -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glColor3f(r,g,b);
  glRasterPos2i(x, y);
  for(p = s, lines = 0; *p; p++) {
    if (*p == '\n') {
      lines++;
      glRasterPos2i(x, y-(lines*18));
    }
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
  }
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void DrawTextBackground(int left,int right,int bottom,int top,float r,float g,float b,float a)
{
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, glutGet(GLUT_WINDOW_WIDTH), 
    0.0, glutGet(GLUT_WINDOW_HEIGHT), -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glColor4f(r,g,b,a);
  glBegin(GL_QUADS);
  glVertex2i(left, bottom);
  glVertex2i(right, bottom);
  glVertex2i(right, top);
  glVertex2i(left, top);
  glEnd();
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void update(int value)
{
  glClear(GL_COLOR_BUFFER_BIT);

  static float simulationTimerSum = 0.f;
  CPUTimer simulationTimer;
  simulationTimer.start();

  if (!pauseSimulation)
  {
    updateSimulation(sTimestep);
    cudaThreadSynchronize();
    simulationTimer.stop();
    simulationTimerSum += simulationTimer.elapsed_sec();
    printf("average simulation time: %f\n",simulationTimerSum / (currentTime/sTimestep));
  }

  switch(currentRenderTarget)
  {
  case RenderTexture:
  case RenderFuel:
  case RenderDensity:
  case RenderTemperature:
  case RenderVelocity:
    pSliceManager->displaySlice(currentSliceToDisplay, currentRenderTarget);
    break;
  case RenderCoarseEngine:
    display();
    break;
  case RenderSingleSlice:
    pRenderer->renderSingleSlice(currentSliceToDisplay);
    break;
  case RenderComposite:
    pRenderer->renderComposite();
    break;
  }

  char drawTime[128];
  sprintf(drawTime, "Time: %f",currentTime);
  char drawVisualization[128];
  sprintf(drawVisualization, "Visualization: %s",currentRenderTargetString.c_str());
  float zIntercept = zBBox.x + ((zBBox.y-zBBox.x)/numSlices)*currentSliceToDisplay;
  char drawZIntercept[128];
  sprintf(drawZIntercept, "Z intercept: %f",zIntercept);

  DrawTextBackground(0,240,0,80,1,1,1,0.5);
  DrawText(10,10,drawTime,1,0,0);
  DrawText(10,30,drawVisualization,1,0,0);
  DrawText(10,50,drawZIntercept,1,0,0);

  glutSwapBuffers();
  glutTimerFunc(0.0001f,update,value);
}

void keyboard (unsigned char key, int x, int y) {
  // x and z move forward and backward through the different visualizations (coarse particles, slice texture, slice density, etc.)
  if (key == 'z' || key == 'x')
  {
    if (key=='z')
      if (--currentRenderTarget < 0)
        currentRenderTarget += numRenderTargets;
    if (key=='x')
      currentRenderTarget = (currentRenderTarget + 1) % numRenderTargets;
    switch(currentRenderTarget)
    {
    case RenderTexture: currentRenderTargetString = "Slice Texture"; break;
    case RenderFuel: currentRenderTargetString = "Slice Fuel"; break;
    case RenderDensity: currentRenderTargetString = "Slice Density"; break;
    case RenderTemperature: currentRenderTargetString = "Slice Temperature"; break;
    case RenderVelocity: currentRenderTargetString = "Slice Velocity"; break;
    case RenderCoarseEngine: currentRenderTargetString = "Coarse Engine"; break;
    case RenderSingleSlice: currentRenderTargetString = "Single Slice"; break;
    case RenderComposite: currentRenderTargetString = "Full Composite"; break;
    }
  }
  // pause simulation
  if (key=='c')
  {
    pauseSimulation = !pauseSimulation;
    pSliceManager->setPauseState(pauseSimulation);
  }
  // rotate up in coarse visualization, nothing in slice visualization
  if (key=='q')
    phi += 10;
  // rotate down in coarse visualization, nothing in slice visualization
  if (key=='e')
    phi -= 10;
  // move forward in coarse visualization, nothing in slice visualization
  if (key=='w')
    cameraDistance -= 10;
  // move backward in coarse visualization, nothing in slice visualization
  if (key=='s')
    cameraDistance += 10;
  // rotate right in coarse visualization, nothing in slice visualization
  if (key=='d')
    theta += 10;
  // rotate left in coarse visualization, nothing in slice visualization
  if (key=='a')
    theta -= 10;
  // change the active slice to visualize forward
  if (key=='r')
    currentSliceToDisplay = (currentSliceToDisplay + 1) % numSlices;
  // change the active slice to visualize backward
  if (key=='f')
    if (--currentSliceToDisplay < 0) currentSliceToDisplay += numSlices;
  // exit
  if (key==27)
    exit(0);

  // RENDERING ADJUSTMENTS
  // print constant values for renderer
  if (key=='p')
    pRenderer->printConstants();
  if (key =='1')
    pRenderer->decreaseTexDensInfluence();
  if (key =='2')
    pRenderer->increaseTexDensInfluence();
  if (key =='3')
    pRenderer->decreaseTexTempInfluence();
  if (key =='4')
    pRenderer->increaseTexTempInfluence();
  if (key =='5')
    pRenderer->decreaseDensityAlphaExp();
  if (key =='6')
    pRenderer->increaseDensityAlphaExp();
  if (key =='7')
    pRenderer->increaseDensityInv();
  if (key =='8')
    pRenderer->decreaseDensityInv();

  if (theta < -360) theta += 360;
  if (theta > 360) theta -= 360;
  if (phi < -360) phi += 360;
  if (phi >360) phi -= 360;
}

void setupProjection(int2 slicePixelDims)
{
  numSliceBytes = sizeof(float)*slicePixelDims.x*slicePixelDims.y;
  numSliceVelocityBytes = sizeof(float2)*slicePixelDims.x*slicePixelDims.y;
  // allocate host memory
  h_sliceMassOutput = (float*) malloc(numSliceBytes);
  h_sliceFuelOutput = (float*) malloc(numSliceBytes);
  h_sliceVelocityOutput = (float2*) malloc(numSliceVelocityBytes);
  // allocate device memory
  cudaMalloc((void**)&d_sliceMassOutput, numSliceBytes);
  cudaMemset(d_sliceMassOutput, 0, numSliceBytes);
  cudaMalloc((void**)&d_sliceFuelOutput, numSliceBytes);
  cudaMemset(d_sliceFuelOutput, 0, numSliceBytes);
  cudaMalloc((void**)&d_sliceVelocityOutput, numSliceVelocityBytes);
  cudaMemset(d_sliceVelocityOutput, 0, numSliceVelocityBytes);
}

void reshape(int w, int h)
{
  glClear(GL_COLOR_BUFFER_BIT);
  window_width = w;
  window_height = h;
  // viewport
  glMatrixMode(GL_PROJECTION);
  glViewport(0, 0, window_width, window_height);
  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 15.0, 200.0);
}

int main(int argc, char* argv[])
{
  srand ( time(NULL) );
  // LOAD SIMULATION SETTINGS
  XMLParser settingsFile("cufire.xml");
  // location of starting particles
  settingsFile.setNewRoot("startingParticleRange");
  float range[2];
  // x range
  settingsFile.getFloat2("xRange",range);
  float2 xRange = make_float2(range[0],range[1]);
  xpos = range[0] + ((range[1]-range[0])/ 2.f);
  // y range
  settingsFile.getFloat2("yRange",range);
  float2 yRange = make_float2(range[0],range[1]);
  ypos = range[0] + ((range[1]-range[0])/ 2.f);
  // z range
  settingsFile.getFloat2("zRange",range);
  float2 zRange = make_float2(range[0],range[1]);
  zpos = range[1]*1.5f;
  settingsFile.resetRoot();
  // get number of starting particles and max number of particles
  int numStartingParticles;
  settingsFile.getInt("numStartingParticles",&numStartingParticles);
  int maxNumParticles;
  settingsFile.getInt("maxNumberParticles",&maxNumParticles);
  settingsFile.getFloat("timestep",&sTimestep);
  settingsFile.getFloat("imageSize",&imageSize);
  settingsFile.getInt("numSlices",&numSlices);
  currentSliceToDisplay = numSlices / 2;
  int jitterAmount;
  settingsFile.getFloat("cameraDistance",&cameraDistance);
  settingsFile.getInt("projectionJitterAmount",&jitterAmount);
  // get bounding box for coarse simulation
  settingsFile.setNewRoot("boundingBox");
  // x range
  settingsFile.getFloat2("xRange",range);
  float2 xBBox = make_float2(range[0],range[1]);
  xpos = range[0] + ((range[1]-range[0])/ 2.f);
  // y range
  settingsFile.getFloat2("yRange",range);
  float2 yBBox = make_float2(range[0],range[1]);
  ypos = range[0] + ((range[1]-range[0])/ 2.f);
  // z range
  settingsFile.getFloat2("zRange",range);
  zBBox = make_float2(range[0],range[1]);
  zpos = range[0] + ((range[1]-range[0])/ 2.f);
  settingsFile.resetRoot();
  // utility values
  float3 gridCenter = make_float3(xBBox.x+(xBBox.y-xBBox.x)/2,
    yBBox.x+(yBBox.y-yBBox.x)/2,
    zBBox.x+(zBBox.y-zBBox.x)/2);
  cameraTarget = gridCenter;
  cameraUp = make_float3(0,1,0);
  theta=0;
  phi=0;
  float3 gridDims = make_float3(xBBox.y-xBBox.x,
    yBBox.y-yBBox.x,
    zBBox.y-zBBox.x);
  float numSlices;
  settingsFile.getFloat("numSlices",&numSlices);
  float projectionDepth = (zBBox.y - zBBox.x ) / (numSlices-1);
  slicePixelDims = make_int2(imageSize,imageSize);
  float2 sliceWorldDims = make_float2(xBBox.y-xBBox.x,
    yBBox.y-yBBox.x);

  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_ACCUM | GLUT_DOUBLE);
  glutInitWindowSize(window_width, window_height);
  //register callbacks
  glutCreateWindow("CUDA Fire Simulation (Coarse Particle Visualization)");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glewInit();
  glClearColor(0.1, 0.1, 0.1, 1.0);
  //set CUDA device
  cudaGLSetGLDevice(0);

  glutTimerFunc(0.0001f, update, 1);

  // add some random particles
  // first get area for random particles
  setupProjection(slicePixelDims);

  pProjection = new OrthographicProjection(gridCenter,gridDims,projectionDepth,slicePixelDims,sliceWorldDims,maxNumParticles,jitterAmount);
  // create particle engine and add particles
  pEngine = new CoarseParticleEngine(maxNumParticles,xBBox,yBBox,zBBox);
  pEngine->addRandomParticle(xRange,yRange,zRange,numStartingParticles);
  pEngine->flushParticles();
  
  
  pSliceManager = new SliceManager("cufire.xml");
  pRenderer = new Renderer(pSliceManager, "cufire.xml");
  
  glutMainLoop();

  return 0;
}