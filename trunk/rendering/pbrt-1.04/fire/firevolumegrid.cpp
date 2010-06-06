// firevolumegrid.cpp*
#include "volume.h"
#include "DensityTemperatureVolume.h"
// FireVolumeGrid Declarations
class FireVolumeGrid : public DensityRegion {
public:
	// FireVolumeGrid Public Methods
	FireVolumeGrid(const Spectrum &sa, const Spectrum &ss, float gg,
	 		const Spectrum &emit, const BBox &e, const Transform &v2w, DensityTemperatureVolume *d);
	~FireVolumeGrid() { delete data; }
	BBox WorldBound() const { return WorldToVolume.GetInverse()(extent); }
	bool IntersectP(const Ray &r, float *t0, float *t1) const {
		Ray ray = WorldToVolume(r);
		return extent.IntersectP(ray, t0, t1);
	}
	float Density(const Point &Pobj) const;
    float Temperature(const Point &Pobj) const;
    Spectrum sigma_a(const Point &p, const Vector &) const {
        printf("%f %f", Density(WorldToVolume(p)), exp(Density(WorldToVolume(p))));
		return exp(Density(WorldToVolume(p))) * sig_a;
	}
    Spectrum Lve(const Point &p, const Vector &) const;
	float D(int x, int y, int z) const {
		x = Clamp(x, 0, data->getSizeInX()-1);
		y = Clamp(y, 0, data->getSizeInY()-1);
		z = Clamp(z, 0, data->getSizeInZ()-1);
		return data->getRawDensityAt(x, y, z);
	}
    float T(int x, int y, int z) const {
		x = Clamp(x, 0, data->getSizeInX()-1);
		y = Clamp(y, 0, data->getSizeInY()-1);
		z = Clamp(z, 0, data->getSizeInZ()-1);
        float t = data->getRawTemperatureAt(x, y, z);
		return t;
	}
    void normalizeXYZ(float xyz[3]) const
    {
        float newXYZ[3];
        for (int i = 0; i < 3; i++)
        {
            newXYZ[i] = 0.0f;
            for (int j = 0;j < 3; j++)
                newXYZ[i] += TnormalizationMat[i*3+j]*xyz[j];
        }
        memcpy(&xyz[0], &newXYZ[0], sizeof(float)*3);
    }
private:
  float RadianceAtLambda(float lambda, float temperature) const;
    void test();
	// VolumeGrid Private Data
	const BBox extent;
    static const int numLambdaSamples = 10;
    static const float constant1;
    static const float constant2;
    static const float lambdaSamples[numLambdaSamples];
    static const float TnormalizationMat[9];
    float CIEWeights[3][numLambdaSamples];
    DensityTemperatureVolume* data;
};

const float FireVolumeGrid::constant1 = 7.4836*1e-16;
const float FireVolumeGrid::constant2 = 1.4388*1e-2;
const float FireVolumeGrid::lambdaSamples[FireVolumeGrid::numLambdaSamples] = {
                            360.0f, 410.0f, 460.0f, 510.0f, 560.0f, 610.0f, 660.0f, 710.0f, 760.0f, 810.0f };
/*const float FireVolumeGrid::TnormalizationMat[9] = { 
            1.82159986e-10, -7.55589605e-11, 4.14127222e-10, 
            -6.91496210e-11, 3.05956638e-10, 1.31545684e-10, 
            8.92502791e-11, -1.54173332e-10, 2.65929718e-09 };*/
const float FireVolumeGrid::TnormalizationMat[9] = { 
            1.75352569e-12, -2.74463007e-13, 9.94998057e-13,
            -3.09319990e-13, 2.33730410e-12, 3.23674602e-13,
            2.03828596e-13, -3.45956678e-13, 7.67706478e-12 };

// FireVolumeGrid Method Definitions
FireVolumeGrid::FireVolumeGrid(const Spectrum &sa,
		const Spectrum &ss, float gg,
 		const Spectrum &emit, const BBox &e,
		const Transform &v2w, DensityTemperatureVolume *d)
	: DensityRegion(sa, ss, gg, emit, v2w), extent(e), data(d) {
    // Normalize
    const float* CIEs[3] = {&(Spectrum::CIE_X[0]), &(Spectrum::CIE_Y[0]), &(Spectrum::CIE_Z[0]) };
    for (int c = 0;c < 3; c++)
    {
        float sum = 0.0f;
        for (int i = 0; i < numLambdaSamples; i++)
        {
            float lambda = lambdaSamples[i];
            int CIEindex = int(lambda) - Spectrum::CIEstart;
            sum += CIEs[c][CIEindex];
        }
//        printf("%d\n", c);
        for (int i = 0; i < numLambdaSamples; i++)
        {
            float lambda = lambdaSamples[i];
            int CIEindex = int(lambda) - Spectrum::CIEstart;
            CIEWeights[c][i] = CIEs[c][CIEindex]/sum;
            printf("%d %f\n", i, CIEWeights[c][i]);
        }
    }
    test();
}
void FireVolumeGrid::test()
{
/*    printf("%d %d %d\n", data->getSizeInX(), data->getSizeInY(), data->getSizeInZ());
    for (int j = 0;j < data->getSizeInY()/3; j++)
        for (int i = data->getSizeInX()/3;i < data->getSizeInX()/3*2; i++)
            for (int k = data->getSizeInZ()/3;k < data->getSizeInZ()/3*2; k++)
                printf("%d %d %d T: %f D: %f\n", i, j, k, data->getRawTemperatureAt(i, j, k), data->getRawDensityAt(i, j, k));*/
/*    for (int i = 0;i < data->getSizeInX(); i++)
        for (int j = 0;j < data->getSizeInY(); j++)
            for (int k = 0;k < data->getSizeInZ(); k++)
                if (data->getRawDensityAt(i, j, k) > 1.2 || data->getRawDensityAt(i, j, k) < 0.0 || 
                    data->getRawTemperatureAt(i, j, k) < 0.0f || data->getRawTemperatureAt(i, j, k) > 2600.f)
                        printf("%f %f\n", data->getRawDensityAt(i, j, k), data->getRawTemperatureAt(i, j, k));*/
    int count = 0;
    int count2 = 0;
    for (int i = 0;i < data->getSizeInX(); i++)
        for (int j = 0;j < data->getSizeInY(); j++)
            for (int k = 0;k < data->getSizeInZ(); k++)
                if (data->getRawTemperatureAt(i, j, k) > 1000.f && data->getRawTemperatureAt(i, j, k) < 2000.f)
                    count++;
                else if (data->getRawTemperatureAt(i, j, k) < 1000.f)
                    count2++;
    printf("P %f\n", float(count)/(512*512*16-count2));
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
	float d00 = Lerp(dx, D(vx, vy, vz),    D(vx+1, vy, vz));
	float d10 = Lerp(dx, D(vx, vy+1, vz),  D(vx+1, vy+1, vz));
	float d01 = Lerp(dx, D(vx, vy, vz+1),  D(vx+1, vy, vz+1));
	float d11 = Lerp(dx, D(vx, vy+1, vz+1),D(vx+1, vy+1, vz+1));
	float d0 = Lerp(dy, d00, d10);
    float d1 = Lerp(dy, d01, d11);
    return Lerp(dz, d0, d1);
}
float FireVolumeGrid::Temperature(const Point &Pobj) const {
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
	float tx = voxx - vx, ty = voxy - vy, tz = voxz - vz;
	// Trilinearly interpolate temperature values to compute local temperature
	float t00 = Lerp(tx, T(vx, vy, vz),    T(vx+1, vy, vz));
	float t10 = Lerp(tx, T(vx, vy+1, vz),  T(vx+1, vy+1, vz));
	float t01 = Lerp(tx, T(vx, vy, vz+1),  T(vx+1, vy, vz+1));
	float t11 = Lerp(tx, T(vx, vy+1, vz+1),T(vx+1, vy+1, vz+1));
	float t0 = Lerp(ty, t00, t10);
    float t1 = Lerp(ty, t01, t11); 
    return Lerp(tz, t0, t1);
}

float FireVolumeGrid::RadianceAtLambda(float lambda, float temperature) const
{
    if (temperature < 1e-6)
        return 0.0;
    lambda *= 1e-9;
    // Planck's formula
    return constant1/(pow(lambda,5)*(exp(constant2/(lambda*temperature))-1));
}

Spectrum FireVolumeGrid::Lve(const Point &p, const Vector &) const {
  float temperature = Temperature(WorldToVolume(p));
  float xyz[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < numLambdaSamples; i++)
  {
    float lambda = lambdaSamples[i];
    float radianceAtLambda = RadianceAtLambda(lambda, temperature);

    for (int c = 0;c < 3; c++)
        xyz[c] += CIEWeights[c][i] * radianceAtLambda;
  }

  normalizeXYZ(xyz);
    
    float rgb[3];
    float x = xyz[0], y=xyz[1], z = xyz[2];
    rgb[0] = 2.5623 * x + (-1.1661) * y + (-0.3962) * z;
    rgb[1] = (-1.0215) * x + 1.9778 * y + 0.0437 * z;
    rgb[2] = 0.0752 * x + (-0.2562) * y + 1.1810 * z;
/*    if (temperature > 1000.0f)
        printf("%f %f %f %f\n", temperature, rgb[0], rgb[1], rgb[2]);*/
    if (temperature > 2000.0f)
        rgb[0] = rgb[1] = rgb[2] = 1.0f;
    return Spectrum(rgb);
/*    float c[3] = {0.9, 0.45, 0.45};
    return Spectrum(c)*Density(WorldToVolume(p));*/
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
