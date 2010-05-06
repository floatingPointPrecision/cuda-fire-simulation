// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, GL
#include <GL/glew.h>
#include <GL/glut.h>

// includes, CUFIRE
#include "CoarseParticle.h"

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;

bool initGL(int argc, char* argv[]);

// rendering callbacks
void display();
void reshape(int w, int h);


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, -3.0);

  glEnable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);

  glutSolidTeapot(1.0);
  glutSwapBuffers();
}

void reshape(int w, int h)
{
  window_width = w;
  window_height = h;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int RunCoarseParticleVisualization(int argc, char* argv[])
{
  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
  if( false == initGL(argc, argv)) {
    return 1;
  }

  // register callbacks
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);

  // start rendering mainloop
  glutMainLoop();

  return 0;
}

bool initGL(int argc, char* argv[])
{
  // first try CUDA test
  TestHelloCUDA();

  // Create GL context
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(window_width, window_height);
  glutCreateWindow("CUDA OpenGL post-processing");

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
  glDisable(GL_DEPTH_TEST);

  // viewport
  glViewport(0, 0, window_width, window_height);

  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glEnable(GL_LIGHT0);
  float red[] = { 1.0, 0.1, 0.1, 1.0 };
  float white[] = { 1.0, 1.0, 1.0, 1.0 };
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0);

  return true;
}
