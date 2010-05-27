#include "SimplexNoise.h"
#include <stdio.h>
#include <vector_types.h>
#include <cstdlib>


SimplexNoise4D::SimplexNoise4D()
{
  m_h_grad = (int4*) malloc(sizeof(int4)*32);
  cudaMalloc((void**)&m_d_grad, sizeof(int4)*32);

  m_h_simplex = (int4*) malloc(sizeof(int4)*64);
  cudaMalloc((void**)&m_d_simplex, sizeof(int4)*64);

  m_h_perm = (int*) malloc(sizeof(int)*512);
  cudaMalloc((void**)&m_d_perm, sizeof(int)*512);

  initializeSimplexValues();
  initializePerumationValues();
  initializeGradVectors();
}

SimplexNoise4D::~SimplexNoise4D()
{
  delete m_h_grad;
  delete m_h_perm;
  cudaFree(m_d_grad);
  cudaFree(m_d_perm);
}

// simplex values taken from http://webstaff.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
void SimplexNoise4D::initializeSimplexValues()
{
  m_h_simplex[0]=make_int4(0,1,2,3);m_h_simplex[1]=make_int4(0,1,3,2);m_h_simplex[2]=make_int4(0,0,0,0);m_h_simplex[3]=make_int4(0,2,3,1);
  m_h_simplex[4]=make_int4(0,0,0,0);m_h_simplex[5]=make_int4(0,0,0,0);m_h_simplex[6]=make_int4(0,0,0,0);m_h_simplex[7]=make_int4(1,2,3,0);
  m_h_simplex[8]=make_int4(0,2,1,3);m_h_simplex[9]=make_int4(0,0,0,0);m_h_simplex[10]=make_int4(0,3,1,2);m_h_simplex[11]=make_int4(0,3,2,1);
  m_h_simplex[12]=make_int4(0,0,0,0);m_h_simplex[13]=make_int4(0,0,0,0);m_h_simplex[14]=make_int4(0,0,0,0);m_h_simplex[15]=make_int4(1,3,2,0);
  m_h_simplex[16]=make_int4(0,0,0,0);m_h_simplex[17]=make_int4(0,0,0,0);m_h_simplex[18]=make_int4(0,0,0,0);m_h_simplex[19]=make_int4(0,0,0,0);
  m_h_simplex[20]=make_int4(0,0,0,0);m_h_simplex[21]=make_int4(0,0,0,0);m_h_simplex[22]=make_int4(0,0,0,0);m_h_simplex[23]=make_int4(0,0,0,0);
  m_h_simplex[24]=make_int4(1,2,0,3);m_h_simplex[25]=make_int4(0,0,0,0);m_h_simplex[26]=make_int4(1,3,0,2);m_h_simplex[27]=make_int4(0,0,0,0);
  m_h_simplex[28]=make_int4(0,0,0,0);m_h_simplex[29]=make_int4(0,0,0,0);m_h_simplex[30]=make_int4(2,3,0,1);m_h_simplex[31]=make_int4(2,3,1,0);
  m_h_simplex[32]=make_int4(1,0,2,3);m_h_simplex[33]=make_int4(1,0,3,2);m_h_simplex[34]=make_int4(0,0,0,0);m_h_simplex[35]=make_int4(0,0,0,0);
  m_h_simplex[36]=make_int4(0,0,0,0);m_h_simplex[37]=make_int4(2,0,3,1);m_h_simplex[38]=make_int4(0,0,0,0);m_h_simplex[39]=make_int4(2,1,3,0);
  m_h_simplex[40]=make_int4(0,0,0,0);m_h_simplex[41]=make_int4(0,0,0,0);m_h_simplex[42]=make_int4(0,0,0,0);m_h_simplex[43]=make_int4(0,0,0,0);
  m_h_simplex[44]=make_int4(0,0,0,0);m_h_simplex[45]=make_int4(0,0,0,0);m_h_simplex[46]=make_int4(0,0,0,0);m_h_simplex[47]=make_int4(0,0,0,0);
  m_h_simplex[48]=make_int4(2,0,1,3);m_h_simplex[49]=make_int4(0,0,0,0);m_h_simplex[50]=make_int4(0,0,0,0);m_h_simplex[51]=make_int4(0,0,0,0);
  m_h_simplex[52]=make_int4(3,0,1,2);m_h_simplex[53]=make_int4(3,0,2,1);m_h_simplex[54]=make_int4(0,0,0,0);m_h_simplex[55]=make_int4(3,1,2,0);
  m_h_simplex[56]=make_int4(2,1,0,3);m_h_simplex[57]=make_int4(0,0,0,0);m_h_simplex[58]=make_int4(0,0,0,0);m_h_simplex[59]=make_int4(0,0,0,0);
  m_h_simplex[60]=make_int4(3,1,0,2);m_h_simplex[61]=make_int4(0,0,0,0);m_h_simplex[62]=make_int4(3,2,0,1);m_h_simplex[63]=make_int4(3,2,1,0);
  cudaMemcpy(m_d_simplex,m_h_simplex,sizeof(int4)*64,cudaMemcpyHostToDevice);
}

// initialize permute values
void SimplexNoise4D::initializePerumationValues()
{
  // create random permutation of first 256 values
  for (int i = 0; i < 256; i++) {
    int j = rand() % (i + 1);
    m_h_perm[i] = m_h_perm[j];
    m_h_perm[j] = i;
  }
  // copy data to second half
  for (int i = 256; i < 512; i++)
    m_h_perm[i] = m_h_perm[i-256];

  cudaMemcpy(m_d_perm, m_h_perm, sizeof(int)*512, cudaMemcpyHostToDevice);
}

void SimplexNoise4D::initializeGradVectors()
{
  m_h_grad[0]=make_int4(0,1,1,1);m_h_grad[1]=make_int4(0,1,1,-1);m_h_grad[2]=make_int4(0,1,-1,1);m_h_grad[3]=make_int4(0,1,-1,-1);
  m_h_grad[4]=make_int4(0,-1,1,1);m_h_grad[5]=make_int4(0,-1,1,-1);m_h_grad[6]=make_int4(0,-1,-1,1);m_h_grad[7]=make_int4(0,-1,-1,-1);
  m_h_grad[8]=make_int4(1,0,1,1);m_h_grad[9]=make_int4(1,0,1,-1);m_h_grad[10]=make_int4(1,0,-1,1);m_h_grad[11]=make_int4(1,0,-1,-1);
  m_h_grad[12]=make_int4(-1,0,1,1);m_h_grad[13]=make_int4(-1,0,1,-1);m_h_grad[14]=make_int4(-1,0,-1,1);m_h_grad[15]=make_int4(-1,0,-1,-1);
  m_h_grad[16]=make_int4(1,1,0,1);m_h_grad[17]=make_int4(1,1,0,-1);m_h_grad[18]=make_int4(1,-1,0,1);m_h_grad[19]=make_int4(1,-1,0,-1);
  m_h_grad[20]=make_int4(-1,1,0,1);m_h_grad[21]=make_int4(-1,1,0,-1);m_h_grad[22]=make_int4(-1,-1,0,1);m_h_grad[23]=make_int4(-1,-1,0,-1);
  m_h_grad[24]=make_int4(1,1,1,0);m_h_grad[25]=make_int4(1,1,-1,0);m_h_grad[26]=make_int4(1,-1,1,0);m_h_grad[27]=make_int4(1,-1,-1,0);
  m_h_grad[28]=make_int4(-1,1,1,0);m_h_grad[29]=make_int4(-1,1,-1,0);m_h_grad[30]=make_int4(-1,-1,1,0);m_h_grad[31]=make_int4(-1,-1,-1,0);
  cudaMemcpy(m_d_grad, m_h_grad, sizeof(int4)*32, cudaMemcpyHostToDevice);
}

__host__ __device__ int fastfloor(float x)
{
  return x > 0? (int)x : (int)x - 1;
}

__host__ __device__ float dot(int4 g, float x, float y, float z, float w)
{
  return g.x*x + g.y*y + g.z*z + g.w*w;
}

__host__ __device__ float noise4D(int4* simplex, int* perm, int4* grad4, float x, float y, float z, float w)
{
  // The skewing and unskewing factors are hairy again for the 4D case
  float F4 = (sqrt(5.0)-1.0)/4.0;
  float G4 = (5.0-sqrt(5.0))/20.0;
  float n0, n1, n2, n3, n4; // Noise contributions from the five corners
  // Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
  float s = (x + y + z + w) * F4; // Factor for 4D skewing
  int i = fastfloor(x + s);
  int j = fastfloor(y + s);
  int k = fastfloor(z + s);
  int l = fastfloor(w + s);
  float t = (i + j + k + l) * G4; // Factor for 4D unskewing
  float X0 = i - t; // Unskew the cell origin back to (x,y,z,w) space
  float Y0 = j - t;
  float Z0 = k - t;
  float W0 = l - t;
  float x0 = x - X0; // The x,y,z,w distances from the cell origin
  float y0 = y - Y0;
  float z0 = z - Z0;
  float w0 = w - W0;
  // For the 4D case, the simplex is a 4D shape I won't even try to describe.
  // To find out which of the 24 possible simplices we're in, we need to
  // determine the magnitude ordering of x0, y0, z0 and w0.
  // The method below is a good way of finding the ordering of x,y,z,w and
  // then find the correct traversal order for the simplex we’re in.
  // First, six pair-wise comparisons are performed between each possible pair
  // of the four coordinates, and the results are used to add up binary bits
  // for an integer index.
  int c1 = (x0 > y0) ? 32 : 0;
  int c2 = (x0 > z0) ? 16 : 0;
  int c3 = (y0 > z0) ? 8 : 0;
  int c4 = (x0 > w0) ? 4 : 0;
  int c5 = (y0 > w0) ? 2 : 0;
  int c6 = (z0 > w0) ? 1 : 0;
  int c = c1 + c2 + c3 + c4 + c5 + c6;
  int i1, j1, k1, l1; // The integer offsets for the second simplex corner
  int i2, j2, k2, l2; // The integer offsets for the third simplex corner
  int i3, j3, k3, l3; // The integer offsets for the fourth simplex corner
  // simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
  // Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
  // impossible. Only the 24 indices which have non-zero entries make any sense.
  // We use a thresholding to set the coordinates in turn from the largest magnitude.
  // The number 3 in the "simplex" array is at the position of the largest coordinate.
  i1 = simplex[c].x>=3 ? 1 : 0;
  j1 = simplex[c].y>=3 ? 1 : 0;
  k1 = simplex[c].z>=3 ? 1 : 0;
  l1 = simplex[c].w>=3 ? 1 : 0;
  // The number 2 in the "simplex" array is at the second largest coordinate.
  i2 = simplex[c].x>=2 ? 1 : 0;
  j2 = simplex[c].y>=2 ? 1 : 0;
  k2 = simplex[c].z>=2 ? 1 : 0;
  l2 = simplex[c].w>=2 ? 1 : 0;
  // The number 1 in the "simplex" array is at the second smallest coordinate.
  i3 = simplex[c].x>=1 ? 1 : 0;
  j3 = simplex[c].y>=1 ? 1 : 0;
  k3 = simplex[c].z>=1 ? 1 : 0;
  l3 = simplex[c].w>=1 ? 1 : 0;
  // The fifth corner has all coordinate offsets = 1, so no need to look that up.
  float x1 = x0 - i1 + G4; // Offsets for second corner in (x,y,z,w) coords
  float y1 = y0 - j1 + G4;
  float z1 = z0 - k1 + G4;
  float w1 = w0 - l1 + G4;
  float x2 = x0 - i2 + 2.0*G4; // Offsets for third corner in (x,y,z,w) coords
  float y2 = y0 - j2 + 2.0*G4;
  float z2 = z0 - k2 + 2.0*G4;
  float w2 = w0 - l2 + 2.0*G4;
  float x3 = x0 - i3 + 3.0*G4; // Offsets for fourth corner in (x,y,z,w) coords
  float y3 = y0 - j3 + 3.0*G4;
  float z3 = z0 - k3 + 3.0*G4;
  float w3 = w0 - l3 + 3.0*G4;
  float x4 = x0 - 1.0 + 4.0*G4; // Offsets for last corner in (x,y,z,w) coords
  float y4 = y0 - 1.0 + 4.0*G4;
  float z4 = z0 - 1.0 + 4.0*G4;
  float w4 = w0 - 1.0 + 4.0*G4;
  // Work out the hashed gradient indices of the five simplex corners
  int ii = i & 255;
  int jj = j & 255;
  int kk = k & 255;
  int ll = l & 255;
  int gi0 = perm[ii+perm[jj+perm[kk+perm[ll]]]] % 32;
  int gi1 = perm[ii+i1+perm[jj+j1+perm[kk+k1+perm[ll+l1]]]] % 32;
  int gi2 = perm[ii+i2+perm[jj+j2+perm[kk+k2+perm[ll+l2]]]] % 32;
  int gi3 = perm[ii+i3+perm[jj+j3+perm[kk+k3+perm[ll+l3]]]] % 32;
  int gi4 = perm[ii+1+perm[jj+1+perm[kk+1+perm[ll+1]]]] % 32;
  // Calculate the contribution from the five corners
  float t0 = 0.6 - x0*x0 - y0*y0 - z0*z0 - w0*w0;
  if(t0<0) n0 = 0.0;
  else {
    t0 *= t0;
    n0 = t0 * t0 * dot(grad4[gi0], x0, y0, z0, w0);
  }
  float t1 = 0.6 - x1*x1 - y1*y1 - z1*z1 - w1*w1;
  if(t1<0) n1 = 0.0;
  else {
    t1 *= t1;
    n1 = t1 * t1 * dot(grad4[gi1], x1, y1, z1, w1);
  }
  float t2 = 0.6 - x2*x2 - y2*y2 - z2*z2 - w2*w2;
  if(t2<0) n2 = 0.0;
  else {
    t2 *= t2;
    n2 = t2 * t2 * dot(grad4[gi2], x2, y2, z2, w2);
  }
  float t3 = 0.6 - x3*x3 - y3*y3 - z3*z3 - w3*w3;
  if(t3<0) n3 = 0.0;
  else {
    t3 *= t3;
    n3 = t3 * t3 * dot(grad4[gi3], x3, y3, z3, w3);
  }
  float t4 = 0.6 - x4*x4 - y4*y4 - z4*z4 - w4*w4;
  if(t4<0) n4 = 0.0;
  else {
    t4 *= t4;
    n4 = t4 * t4 * dot(grad4[gi4], x4, y4, z4, w4);
  }
  // Sum up and scale the result to cover the range [-1,1]
  return 27.0 * (n0 + n1 + n2 + n3 + n4);
}

__global__ void updateNoiseField(float* fieldValues, int4* simplex, int* perm, int4* grad4, int z, float time, float scale, int numElements, int dim)
{
  __shared__ char shared[sizeof(int4)*32 + sizeof(int4)*64 + sizeof(int)*512];
  int4* sharedSimplex = (int4*) (&shared[0]);
  int* sharedPerm = (int*) (&shared[64*sizeof(int4)]);
  int4* sharedGrad4 = (int4*) (&shared[64*sizeof(int4)+512*sizeof(int)]);
  int index = blockDim.x*blockIdx.x+threadIdx.x;
  if (index >= numElements)
    return;
  if (threadIdx.x < 32)
    sharedGrad4[threadIdx.x] = grad4[threadIdx.x];
  if (threadIdx.x < 64)
    sharedSimplex[threadIdx.x] = simplex[threadIdx.x];
  if (threadIdx.x < 512)
    sharedPerm[threadIdx.x] = perm[threadIdx.x];
  __syncthreads();
  fieldValues[index] += noise4D(simplex, perm, grad4, scale * (index/dim), scale * (index%dim), scale * z, scale * time);
}


void SimplexNoise4D::updateNoise(float* scalarField, float z, float time, float scale, int numElements, int dim)
{
  int blockSize = 512;
  int gridSize = (numElements + blockSize - 1) / blockSize;
  updateNoiseField<<<gridSize,blockSize>>>(scalarField, m_d_simplex, m_d_perm, m_d_grad, z, time, scale, numElements, dim);
}