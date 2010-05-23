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
#include <string.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <GL/glew.h>
#include <GL/glut.h>

// includes, CUDA
#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include "ocuutil/timer.h"

// includes, CUFIRE
#include "XMLParser.h"
#include "CoarseParticleEngine.h"
#include "CoarseParticleVisualization.h"
#include "Projection.h"
#include "3DNavierStokes.h"
#include "Bitmap.hpp"

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;

using namespace cufire;

CoarseParticleEngine* pEngine;
OrthographicProjection* pProjection;
float sTimestep;

float4* d_sliceOutput;
float4* h_sliceOutput;
int numSliceBytes;
int2 slicePixelDims;
//angle of rotation
float xpos = 32, ypos = 32, zpos = 90, xrot = 0, yrot = 0, angle=0.0;


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
  printf("\n\n     NEW TIME STEP\n");
  // move VBO from OpenGL context to CUDA context
  pEngine->enableCUDAVbo();
  // update coarse simulation particles
  pEngine->advanceSimulation(1/60.f);

  // project coarse particles onto slices
  CPUTimer projectionTimer;
  projectionTimer.start();
  pProjection->setParticles(pEngine->getParticleBegins(), pEngine->getNumParticles());
  for (int i = 0; i < 1; i++)
  {
    cudaMemset(d_sliceOutput,0,numSliceBytes);
    // perform actual projection for s lice # i
    pProjection->setSliceInformation(32.f, d_sliceOutput);
    pProjection->execute();
    // copy output as image
    cudaMemcpy(h_sliceOutput, d_sliceOutput, numSliceBytes, cudaMemcpyDeviceToHost);
    BitmapWriter newBitmap(slicePixelDims.x,slicePixelDims.y);
    int x,y,width=slicePixelDims.x,height=slicePixelDims.y;
    for(x=0;x<width;++x)
    {
      for(y=0;y<height;++y)
      {
        float4 currentPixel = h_sliceOutput[y*width+x];
        newBitmap.setValue(x,y,char(currentPixel.x),char(currentPixel.y),char(currentPixel.z));
      }
    }
    std::string fileName;
    std::stringstream ss(std::stringstream::in | std::stringstream::out);
    ss << "outputBitmap" << i << ".bmp";
    fileName = ss.str();
    newBitmap.flush(fileName.c_str());
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

void display()
{
  updateSimulation(sTimestep);
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslated(-xpos,-ypos,0); // translation
  glRotatef(xrot,1.0,0.0,0.0); // x rotation
  glRotatef(yrot,0.0,1.0,0.0); // y rotation
  glTranslated(0,0,-zpos);
   
  // draw bounding box
  drawCube(make_float3(0,0,0), make_float3(64,64,64));
  pEngine->render();

  glutSwapBuffers();
}

void reshape(int w, int h)
{
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

bool initGL(int argc, char* argv[])
{
  // Create GL context
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(window_width, window_height);
  glutCreateWindow("CUDA Fire Simulation (Particle Visualization)");

  // initialize necessary OpenGL extensions
  glewInit();
  if (! glewIsSupported(
    "GL_VERSION_2_0 " 
    "GL_ARB_pixel_buffer_object "
    "GL_EXT_framebuffer_object "
    )) {
      fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
      fflush(stderr);
      return false;
  }

  // default initialization
  glClearColor(0.5, 0.5, 0.5, 1.0);
  //glEnable(GL_DEPTH_TEST);

  /*glEnable(GL_LIGHT0);
  float red[] = { 1.0, 0.1, 0.1, 1.0 };
  float white[] = { 1.0, 1.0, 1.0, 1.0 };
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0);*/

  return true;
}

void timer(int value)
{
  display();
  glutTimerFunc(0.0001f,timer,value);
}

// based on OpenGL Camera Tutorial at http://www.swiftless.com/tutorials/opengl/camera.html
void keyboard (unsigned char key, int x, int y) {
  if (key=='q')
  {
    xrot += 10;
    if (xrot >360) xrot -= 360;
  }

  if (key=='z')
  {
    xrot -= 10;
    if (xrot < -360) xrot += 360;
  }

  if (key=='w')
  {
    float xrotrad, yrotrad;
    yrotrad = (yrot / 180 * 3.141592654f);
    xrotrad = (xrot / 180 * 3.141592654f);
    xpos += float(sin(yrotrad)) ;
    zpos -= float(cos(yrotrad)) ;
    ypos -= float(sin(xrotrad)) ;
  }

  if (key=='s')
  {
    float xrotrad, yrotrad;
    yrotrad = (yrot / 180 * 3.141592654f);
    xrotrad = (xrot / 180 * 3.141592654f);
    xpos -= float(sin(yrotrad));
    zpos += float(cos(yrotrad)) ;
    ypos += float(sin(xrotrad));
  }

  if (key=='d')
  {
    yrot += 10;
    if (yrot >360) yrot -= 360;
  }

  if (key=='a')
  {
    yrot -= 10;
    if (yrot < -360)yrot += 360;
  }
  if (key==27)
  {
    exit(0);
  }
}

int RunCoarseParticleVisualization(int argc, char* argv[])
{
  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
  if(initGL(argc, argv) == false) {
    return 1;
  }
  // set CUDA device
  cudaGLSetGLDevice (cutGetMaxGflopsDeviceId());

  // register callbacks
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutTimerFunc(0.0001f, timer, 1);
  glutKeyboardFunc(keyboard);

  // add some random particles
  // first get area for random particles
  srand ( time(NULL) );
  XMLParser settingsFile("ParticleSettings.xml");
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

  // get bounding box for particles
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
  zpos = range[1]*2.0f;
  settingsFile.resetRoot();

 
  float3 gridCenter = make_float3(xBBox.x+(xBBox.y-xBBox.x)/2,
    yBBox.x+(yBBox.y-yBBox.x)/2,
    zBBox.x+(zBBox.y-zBBox.x)/2);
  float3 gridDims = make_float3(xBBox.y-xBBox.x,
    yBBox.y-yBBox.x,
    zBBox.y-zBBox.x);
  float projectionDepth = 2.0f;
  slicePixelDims = make_int2(300,300);
  float2 sliceWorldDims = make_float2(xBBox.y-xBBox.x,
    yBBox.y-yBBox.x);
  numSliceBytes = sizeof(float4)*slicePixelDims.x*slicePixelDims.y;
  h_sliceOutput = (float4*) malloc(numSliceBytes);
  cudaMalloc((void**)&d_sliceOutput, numSliceBytes);
  cudaMemset(d_sliceOutput,0, numSliceBytes);
  pProjection = new OrthographicProjection(gridCenter,gridDims,projectionDepth,slicePixelDims,sliceWorldDims);

  // create particle engine
  pEngine = new CoarseParticleEngine(maxNumParticles,xBBox,yBBox,zBBox);
  pEngine->addRandomParticle(xRange,yRange,zRange,numStartingParticles);
  pEngine->flushParticles();


  // start rendering mainloop

  glutMainLoop();

  return 0;
}