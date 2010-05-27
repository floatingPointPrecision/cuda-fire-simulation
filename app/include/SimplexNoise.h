#pragma once


class SimplexNoise4D
{
public:
  SimplexNoise4D();
  ~SimplexNoise4D();

  void updateNoise(float* scalarField, float z, float time, float scale, int numElements, int dim);

protected:
  void initializeSimplexValues();
  void initializePerumationValues();
  void initializeGradVectors();

  int4* m_h_grad;
  int4* m_d_grad;

  int4* m_h_simplex;
  int4* m_d_simplex;

  int* m_h_perm;
  int* m_d_perm;

};