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
#include "fluidsGL.h"

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;

using namespace cufire;

int enableCoarseVisualization;

int numRenderTargets = 6;
enum RenderTarget
{
  RenderTexture = 0,
  RenderDensity,
  RenderFuel,
  RenderTemperature,
  RenderVelocity,
  RenderCoarseEngine
};
std::string currentRenderTargetString = "Coarse Engine";

int currentRenderTarget = RenderCoarseEngine;
CoarseParticleEngine* pEngine;
OrthographicProjection* pProjection;

float sTimestep;
float currentTime = 0;

float* d_sliceMassOutput;
float* d_sliceFuelOutput;
float2* d_sliceVelocityOutput;

float* h_sliceMassOutput;
float* h_sliceFuelOutput;
float2* h_sliceVelocityOutput;
int numSliceBytes, numSliceVelocityBytes;
int2 slicePixelDims;
float imageSize;
//angle of rotation
float xpos = 32, ypos = 32, zpos = 90, xrot = 0, yrot = 0, angle=0.0;
float3 cameraTarget, cameraUp;
float theta, phi, cameraDistance;


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
  printf("\n\n     NEW TIME STEP\n");
  // move VBO from OpenGL context to CUDA context
  pEngine->enableCUDAVbo();
  // update coarse simulation particles
  pEngine->advanceSimulation(dt);

  // project coarse particles onto slices
  CPUTimer projectionTimer;
  projectionTimer.start();
  pProjection->setParticles(pEngine->getParticleBegins(), pEngine->getNumParticles());

  BitmapWriter massImage(slicePixelDims.x,slicePixelDims.y);
  BitmapWriter fuelImage(slicePixelDims.x,slicePixelDims.y);
  BitmapWriter velocityImage(slicePixelDims.x,slicePixelDims.y);

  for (int i = 0; i < 1; i++)
  {
    float zIntercept = 32.f;
    cutilSafeCall(cudaMemset(d_sliceMassOutput,0,numSliceBytes));
    cutilSafeCall(cudaMemset(d_sliceFuelOutput,0,numSliceBytes));
    cutilSafeCall(cudaMemset(d_sliceVelocityOutput,0,numSliceVelocityBytes));
    // perform actual projection for slice # i
    pProjection->execute(zIntercept, d_sliceMassOutput, d_sliceFuelOutput, d_sliceVelocityOutput);
    // copy output as image
    cutilSafeCall(cudaMemcpy(h_sliceMassOutput, d_sliceMassOutput, numSliceBytes, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(h_sliceFuelOutput, d_sliceFuelOutput, numSliceBytes, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(h_sliceVelocityOutput, d_sliceVelocityOutput, numSliceVelocityBytes, cudaMemcpyDeviceToHost));

    CPUTimer fluid2DTimer;
    fluid2DTimer.start();
    replaceVelocityField(d_sliceVelocityOutput);
    dissipateDensity(dt);
    dissipateFuel(dt);
    coolTemperature(dt);
    contributeSlices(d_sliceMassOutput, d_sliceFuelOutput);
    simulateFluids(dt);
    addTextureDetail(currentTime, zIntercept);
    enforveVelocityIncompressibility(dt);
    fluid2DTimer.stop();
    printf("fluid 2D time: %f\n", fluid2DTimer.elapsed_sec());
  }

  projectionTimer.stop();
  printf("Projection time: %f\n", projectionTimer.elapsed_sec());

  // move VBO back to OpenGL
  pEngine->disableCUDAVbo();
}

void drawCube(float3 lowerLeftFront, float3 upperRightBack)
{
  glEnable(GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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

#define M_PI 3.14159265f
void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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

void reshape(int w, int h)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  window_width = w;
  window_height = h;
  // viewport
  glMatrixMode(GL_PROJECTION);
  glViewport(0, 0, window_width, window_height);
  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //glOrtho(-40, 40, -40, 40, 1, 80);

  gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 15.0, 200.0);
}

void timer2(int value)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  // update coarse particle simulation
  updateSimulation(sTimestep);
  //  RenderTexture = 0,
  //RenderDensity,
  //RenderFuel,
  //RenderTemperature,
  //RenderVelocity,
  //RenderCoarseEngine
  switch(currentRenderTarget)
  {
  case RenderTexture:
  case RenderFuel:
  case RenderDensity:
  case RenderTemperature:
  case RenderVelocity:
    displaySlice(currentRenderTarget);
    break;
  case RenderCoarseEngine:
    display();
    break;
  }

  char drawTime[128];
  sprintf(drawTime, "Time: %f",currentTime);
  char drawVisualization[128];
  sprintf(drawVisualization, "Visualization: %s",currentRenderTargetString.c_str());
  DrawText(10,10,drawTime,1,0,0);
  DrawText(10,30,drawVisualization,1,0,0);

  glutSwapBuffers();
  glutTimerFunc(0.0001f,timer2,value);
}

// based on OpenGL Camera Tutorial at http://www.swiftless.com/tutorials/opengl/camera.html
void keyboard (unsigned char key, int x, int y) {
  if (key == 'z' || key == 'x')
  {
    if (key=='z')
    {
      currentRenderTarget--;
      if (currentRenderTarget == -1)
        currentRenderTarget = numRenderTargets - 1;
    }
    if (key=='x')
    {
      currentRenderTarget = (currentRenderTarget + 1) % numRenderTargets;
    }
    switch(currentRenderTarget)
    {
    case RenderTexture: currentRenderTargetString = "Slice Texture"; break;
    case RenderFuel: currentRenderTargetString = "Slice Fuel"; break;
    case RenderDensity: currentRenderTargetString = "Slice Density"; break;
    case RenderTemperature: currentRenderTargetString = "Slice Temperature"; break;
    case RenderVelocity: currentRenderTargetString = "Slice Velocity"; break;
    case RenderCoarseEngine: currentRenderTargetString = "Coarse Engine"; break;
    }
  }
  if (key=='q')
  {
    phi += 10;
    if (phi >360) phi -= 360;
  }

  if (key=='e')
  {
    phi -= 10;
    if (phi < -360) phi += 360;
  }

  if (key=='w')
  {
    cameraDistance -= 10;
  }

  if (key=='s')
  {
    cameraDistance += 10;
  }

  if (key=='d')
  {
    theta += 10;
    if (theta >360) theta -= 360;
  }

  if (key=='a')
  {
    theta -= 10;
    if (theta < -360) theta += 360;
  }
  if (key==27)
  {
    exit(0);
  }
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

int main(int argc, char* argv[])
{
  srand ( time(NULL) );
  // LOAD SIMULATION SETTINGS
  XMLParser settingsFile("ParticleSettings.xml");
  settingsFile.getInt("coarseVisualization",&enableCoarseVisualization);
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
  float2 zBBox = make_float2(range[0],range[1]);
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
  float projectionDepth = 2.0f;
  slicePixelDims = make_int2(imageSize,imageSize);
  float2 sliceWorldDims = make_float2(xBBox.y-xBBox.x,
    yBBox.y-yBBox.x);

  // enable either coarse particle visualization or slice simulation
  if (enableCoarseVisualization)
  {
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    //register callbacks
    glutCreateWindow("CUDA Fire Simulation (Coarse Particle Visualization)");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glewInit();
    glClearColor(0.5, 0.5, 0.5, 1.0);
    //set CUDA device
    cudaGLSetGLDevice(0);
  }
  else
  {
    setupSliceVisualization(argc,argv);
  }
  glutTimerFunc(0.0001f, timer2, 1);

  // add some random particles
  // first get area for random particles
  setupProjection(slicePixelDims);

  pProjection = new OrthographicProjection(gridCenter,gridDims,projectionDepth,slicePixelDims,sliceWorldDims,maxNumParticles,jitterAmount);
  // create particle engine
  pEngine = new CoarseParticleEngine(maxNumParticles,xBBox,yBBox,zBBox);
  pEngine->addRandomParticle(xRange,yRange,zRange,numStartingParticles);
  pEngine->flushParticles();

  setupSliceSimulation();
  //pSliceRefiner = new SliceRefiner(imageSize, argc, argv);
  // start rendering mainloop

  glutMainLoop();

  return 0;
}