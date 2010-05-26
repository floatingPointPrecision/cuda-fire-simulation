#include "SliceRefiner.h"

using namespace cufire;

// CUFFT plan handle
    cufftHandle planr2c;
    cufftHandle planc2r;
    cData *vxfield;
    cData *vyfield;
    cData *hvfield;
    cData *dvfield;
    int wWidth;
    int wHeight;

    int clicked;
    int fpsCount;
    int fpsLimit;
    unsigned int timer;

    // Particle data
    GLuint vbo;                 // OpenGL vertex buffer object
    cData *particles; // particle positions in host memory
    int lastx, lasty;

    // Texture pitch
    size_t tPitch; // Now this is compatible with gcc in 64-bit


void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r) { 

    dim3 tids(2*r+1, 2*r+1);
    
    addForces_k<<<1, tids>>>(v, dx, dy, spx, spy, fx, fy, r, tPitch);
    cutilCheckMsg("addForces_k failed.");
}

void advectVelocity(cData *v, float *vx, float *vy,
                    int dx, int pdx, int dy, float dt) { 
    
    dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

    dim3 tids(TIDSX, TIDSY);

    updateTexture(v, DIM*sizeof(cData), DIM, tPitch);
    advectVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, dt, TILEY/TIDSY);

    cutilCheckMsg("advectVelocity_k failed.");
}

void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt,
                    float visc) { 
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
                    int dx, int pdx, int dy) { 

    dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

    dim3 tids(TIDSX, TIDSY);

    updateVelocity_k<<<grid, tids>>>(v, vx, vy, dx, pdx, dy, TILEY/TIDSY, tPitch);
    cutilCheckMsg("updateVelocity_k failed.");
}

void advectParticles(GLuint buffer, cData *v, int dx, int dy, float dt) {
    
    dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

    dim3 tids(TIDSX, TIDSY);

    cData *p;
    cudaGLMapBufferObject((void**)&p, buffer);
    cutilCheckMsg("cudaGLMapBufferObject failed");
   
    advectParticles_k<<<grid, tids>>>(p, v, dx, dy, dt, TILEY/TIDSY, tPitch);
    cutilCheckMsg("advectParticles_k failed.");
    
    cudaGLUnmapBufferObject(buffer);
    cutilCheckMsg("cudaGLUnmapBufferObject failed");
}

void sliceDisplay(void) {  
   cutilCheckError(cutStartTimer(timer));  
    
   // simulate fluid
   advectVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM, DT);
   diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS);
   updateVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM);
   advectParticles(vbo, dvfield, DIM, DIM, DT);
   
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

void initParticles(cData *p, int dx, int dy) 
{
    int i, j;
    for (i = 0; i < dy; i++) {
        for (j = 0; j < dx; j++) {
            p[i*dx+j].x = ((j+0.5)/dx) + 
                          (rand() / (float)RAND_MAX - 0.5f) / dx;
            p[i*dx+j].y = ((i+0.5)/dy) + 
                          (rand() / (float)RAND_MAX - 0.5f) / dy;
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

            cudaGLUnregisterBufferObject(vbo);
            cutilCheckMsg("cudaGLUnregisterBufferObject failed");
    
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
            glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS, 
                            particles, GL_DYNAMIC_DRAW_ARB);
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

            cudaGLRegisterBufferObject(vbo);
            cutilCheckMsg("cudaGLRegisterBufferObject failed");
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

SliceRefiner::SliceRefiner(int domainSquareSize,int argc, char* argv[])
{


  vxfield = NULL;
  vyfield = NULL;
  hvfield = NULL;
  dvfield = NULL;
  wWidth = max(512,DIM);
  wHeight = max(512,DIM);
  clicked = 0;
  fpsCount = 0;
  fpsLimit = 1;
  vbo = 0;                 // OpenGL vertex buffer object
  particles = NULL; // particle positions in host memory
  lastx = 0;
  lasty = 0;
  tPitch = 0; // Now this is compatible with gcc in 64-bit

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

  // Allocate and initialize host data
  cutCreateTimer(&timer);
  cutResetTimer(timer);
  hvfield = (cData*)malloc(sizeof(cData) * DS);
  memset(hvfield, 0, sizeof(cData) * DS);
  // Allocate and initialize device data
  cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(cData)*DIM, DIM);
  cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, 
    cudaMemcpyHostToDevice); 
  // Temporary complex velocity field data     
  cudaMalloc((void**)&vxfield, sizeof(cData) * PDS);
  cudaMalloc((void**)&vyfield, sizeof(cData) * PDS);

  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    printf("CUDA ERROR!\n");
    exit(-1);
  }

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

  cudaGLRegisterBufferObject(vbo);

  err = cudaGetLastError();
  if( cudaSuccess != err) {
    printf("CUDA ERROR!\n");
    exit(-1);
  }

  glutMainLoop();
}

SliceRefiner::~SliceRefiner()
{
}

void SliceRefiner::setSliceInformation()
{
}

void SliceRefiner::updateSimulation(float dt)
{
}

