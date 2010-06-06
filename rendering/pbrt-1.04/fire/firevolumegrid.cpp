
/*
    pbrt source code Copyright(c) 1998-2007 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    pbrt is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.  Note that the text contents of
    the book "Physically Based Rendering" are *not* licensed under the
    GNU GPL.

    pbrt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 */

// volumegrid.cpp*
#include "volume.h"
#include "DensityTemperatureVolume.h"
// VolumeGrid Declarations
class FireVolumeGrid : public DensityRegion {
public:
	// VolumeGrid Public Methods
	FireVolumeGrid(const Spectrum &sa, const Spectrum &ss, float gg,
	 		const Spectrum &emit, const BBox &e, const Transform &v2w, DensityTemperatureVolume *d);
	~FireVolumeGrid() { delete data; }
	BBox WorldBound() const { return WorldToVolume.GetInverse()(extent); }
	bool IntersectP(const Ray &r, float *t0, float *t1) const {
		Ray ray = WorldToVolume(r);
		return extent.IntersectP(ray, t0, t1);
	}
	float Density(const Point &Pobj) const;
  Spectrum Lve(const Point &p, const Vector &) const;
	float D(int x, int y, int z) const {
		x = Clamp(x, 0, data->getSizeInX()-1);
		y = Clamp(y, 0, data->getSizeInY()-1);
		z = Clamp(z, 0, data->getSizeInZ()-1);
		return data->getDensityAt(x, y, z);
	}
private:
  float RadianceAtLambda(float lambda, float temperature) const;
    void test();
	// VolumeGrid Private Data
	const BBox extent;
    static const int numLambdaSamples = 5;
    static const float constant1;
    static const float constant2;
    static const float lambdaSamples[numLambdaSamples];
    DensityTemperatureVolume* data;
};

const float FireVolumeGrid::constant1 = 7.4836*10e-16;
const float FireVolumeGrid::constant2 = 1.4388*10e-2;
const float FireVolumeGrid::lambdaSamples[FireVolumeGrid::numLambdaSamples] = {
                            400.0f, 500.0f, 600.0f, 700.0f, 800.0f };

// VolumeGrid Method Definitions
FireVolumeGrid::FireVolumeGrid(const Spectrum &sa,
		const Spectrum &ss, float gg,
 		const Spectrum &emit, const BBox &e,
		const Transform &v2w, DensityTemperatureVolume *d)
	: DensityRegion(sa, ss, gg, emit, v2w), extent(e), data(d) {
}
void FireVolumeGrid::test()
{
    printf("%d %d %d\n", data->getSizeInX(), data->getSizeInY(), data->getSizeInZ());
    for (int i = 0;i < data->getSizeInX(); i++)
        for (int j = 0;j < data->getSizeInY(); j++)
            for (int k = 0;k < data->getSizeInZ(); k++)
                printf("%f\n", data->getDensityAt(i, j, k));
}
float FireVolumeGrid::Density(const Point &Pobj) const {
    if (!extent.Inside(Pobj)) return 0;
	// Compute voxel coordinates and offsets for _Pobj_
	float voxx = (Pobj.x - extent.pMin.x) /
		(extent.pMax.x - extent.pMin.x) * data->getSizeInX() - .5f;
	float voxy = (Pobj.y - extent.pMin.y) /
		(extent.pMax.y - extent.pMin.y) * data->getSizeInY() - .5f;
	float voxz = (Pobj.z - extent.pMin.z) /
		(extent.pMax.z - extent.pMin.z) * data->getSizeInZ() - .5f;
	int vx = Floor2Int(voxx);
	int vy = Floor2Int(voxy);
	int vz = Floor2Int(voxz);
	float dx = voxx - vx, dy = voxy - vy, dz = voxz - vz;
	// Trilinearly interpolate density values to compute local density
	float d00 = Lerp(dx, D(vx, vy, vz),     D(vx+1, vy, vz));
	float d10 = Lerp(dx, D(vx, vy+1, vz),  D(vx+1, vy+1, vz));
	float d01 = Lerp(dx, D(vx, vy, vz+1),  D(vx+1, vy, vz+1));
	float d11 = Lerp(dx, D(vx, vy+1, vz+1),D(vx+1, vy+1, vz+1));
	float d0 = Lerp(dy, d00, d10);
    float d1 = Lerp(dy, d01, d11);
    return Lerp(dz, d0, d1);
}

float FireVolumeGrid::RadianceAtLambda(float lambda, float temperature) const
{
    return constant1/(pow(lambda,5)*(exp(constant2/(lambda*temperature))-1));
}

Spectrum FireVolumeGrid::Lve(const Point &p, const Vector &) const {
//  float density = Density(WorldToVolume(p));
  float temperature = 1700.f;
  float xyz[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < numLambdaSamples; i++)
  {
    float lambda = lambdaSamples[i];
    float radianceAtLambda = RadianceAtLambda(lambda, temperature);

    int CIEindex = int(lambda) - Spectrum::CIEstart;
    xyz[0] += Spectrum::CIE_X[CIEindex] * radianceAtLambda;    
    xyz[1] += Spectrum::CIE_Y[CIEindex] * radianceAtLambda;    
    xyz[2] += Spectrum::CIE_Z[CIEindex] * radianceAtLambda;
  }
  
  return FromXYZ(xyz[0], xyz[1], xyz[2]);
}

extern "C" DLLEXPORT VolumeRegion *CreateVolumeRegion(const Transform &volume2world,
                                                      const ParamSet &params) {
                                                        // Initialize common volume region parameters
	Spectrum sigma_a = params.FindOneSpectrum("sigma_a", 0.);
	Spectrum sigma_s = params.FindOneSpectrum("sigma_s", 0.);
	float g = params.FindOneFloat("g", 0.);
	Spectrum Le = params.FindOneSpectrum("Le", 0.);
	Point p0 = params.FindOnePoint("p0", Point(0,0,0));
	Point p1 = params.FindOnePoint("p1", Point(1,1,1));
/*	int nitems;
	const float *data = params.FindFloat("density", &nitems);
	if (!data) {
		Error("No \"density\" values provided for volume grid?");
		return NULL;
	}
	int nx = params.FindOneInt("nx", 1);
	int ny = params.FindOneInt("ny", 1);
	int nz = params.FindOneInt("nz", 1);
	if (nitems != nx*ny*nz) {
		Error("VolumeGrid has %d density values but nx*ny*nz = %d",
			nitems, nx*ny*nz);
		return NULL;
	}*/
    string simFileName = params.FindOneString("simFile", "sliceData.bin");
    DensityTemperatureVolume* data = new DensityTemperatureVolume();
    if (!data->load(simFileName.c_str()))
    {
        Error("FireVolumeGrid failed to load simulation file %s\n", simFileName.c_str());
        delete data;
        return NULL;
    }
    printf("Loaded %s\n", simFileName.c_str());
	return new FireVolumeGrid(sigma_a, sigma_s, g, Le, BBox(p0, p1),
		volume2world, data);
}
