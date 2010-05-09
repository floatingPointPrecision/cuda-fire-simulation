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
#include <GL/glew.h>
#include <GL/glut.h>

// includes, CUDA
#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

// includes, CUFIRE
#include "CoarseParticleEngine.h"
#include "CoarseParticleVisualization.h"

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;

using namespace cufire;

CoarseParticleEngine* pEngine;

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

void display()
{
  pEngine->advanceSimulation(1/60.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, -10.0);

  glEnable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);

  //glutSolidTeapot(1.0);
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
  gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
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
  glEnable(GL_DEPTH_TEST);

  glEnable(GL_LIGHT0);
  float red[] = { 1.0, 0.1, 0.1, 1.0 };
  float white[] = { 1.0, 1.0, 1.0, 1.0 };
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0);

  return true;
}

void timer(int value)
{
  display();
  glutTimerFunc(1000/60.f,timer,value);
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
  glutTimerFunc(1000/60.f, timer, 1);

  // add some random particles
  srand ( time(NULL) );
  pEngine = new CoarseParticleEngine();
  float2 xRange = make_float2(-5, 5);
  float2 yRange = make_float2(-5, 5);
  float2 zRange = make_float2(0, 5);
  pEngine->addRandomParticle(xRange,yRange,zRange,1<<17);
  pEngine->flushParticles();


  // start rendering mainloop

  glutMainLoop();

  return 0;
}