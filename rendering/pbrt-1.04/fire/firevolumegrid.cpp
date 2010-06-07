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
    void CalcXYZ(float temperature, float xyz[3]) const;
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
/*const float FireVolumeGrid::TnormalizationMat[9] = { 
            8.10306604e-11, -2.91192204e-11, 1.48607220e-10,
            -2.79038251e-11, 1.31633769e-10, 4.73688255e-11,
            3.17980423e-11, -5.47962386e-11, 9.69330402e-10 };*/
/*const float FireVolumeGrid::TnormalizationMat[9] = { 
            2.2493871, -1.04837468, 3.96512291,
            -0.11514529, 2.63877031, 0.02335136,
            0., 0., 21.88440292 };*/
const float FireVolumeGrid::TnormalizationMat[9] = { 
            8.94637901e-11, -4.16965012e-11, 1.57702924e-10,
            -4.57961804e-12, 1.04950541e-10, 9.28742452e-13,
            0., 0., 8.70397820e-10 };

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

inline void normalizeVec3(float vec[3])
{
    float sum = vec[0] + vec[1] + vec[2];
    if (sum > 1e-9)
    {
        for (int i = 0;i < 3; i++)
            vec[i] /= sum;
    }
}

Spectrum RGBFromXYZ(float xyz[3])
{
    float rgb[3];
    rgb[0] = 3.240479*xyz[0] - 1.537150*xyz[1] - 0.498535*xyz[2];
    rgb[1] = -0.969256*xyz[0] + 1.875991*xyz[1] + 0.041556*xyz[2]; 
    rgb[2] = 0.055648*xyz[0] -0.204043*xyz[1] + 1.057311*xyz[2];
    return Spectrum(rgb);
}

void FireVolumeGrid::CalcXYZ(float temperature, float xyz[3]) const
{
    memset(&xyz[0], 0, sizeof(float)*3);
/*    if (temperature >= 1666)
    {
        float temperatureSqr = temperature*temperature;
        float temperatureCub = temperatureSqr*temperature;
        xyz[0] = -0.2661239*1e9/temperatureCub - 0.2343580*1e6/temperatureSqr + 0.8776956*1e3/temperature + 0.179910;
        float xcSqr = xyz[0]*xyz[0];
        float xcCub = xcSqr*xyz[0];
        xyz[1] = -1.1063814*xcCub -1.34811020*xcSqr + 2.18555832*xyz[0] - 0.20219683;
        xyz[2] = 1.0f-xyz[1]-xyz[0];
        return;
    }*/
    for (int i = 0; i < numLambdaSamples; i++)
    {
        float lambda = lambdaSamples[i];
        float radianceAtLambda = RadianceAtLambda(lambda, temperature);

        for (int c = 0;c < 3; c++)
            xyz[c] += CIEWeights[c][i] * radianceAtLambda;
    }
}

Spectrum FireVolumeGrid::Lve(const Point &p, const Vector &) const {
    float temperature = Temperature(WorldToVolume(p));
    float xyz[3];
    CalcXYZ(temperature, xyz);
  
    float prevXYZ[3] = {xyz[0], xyz[1], xyz[2]};
    normalizeVec3(prevXYZ);
//    normalizeVec3(xyz);
    normalizeXYZ(xyz);
    
    if (prevXYZ[0] > 1e-6)
    {
#define DIS 0
#if DIS
        printf("Temp %f \n", temperature);
        printf("%f %f %f\n", prevXYZ[0], prevXYZ[1], prevXYZ[2]);
        printf("%f %f %f\n", xyz[0], xyz[1], xyz[2]);
#endif
//        Spectrum c = RGBFromXYZ(prevXYZ);
//        Spectrum c2 = RGBFromXYZ(xyz);
        float ratioYX = prevXYZ[0]/prevXYZ[1];
        float ratioZX = prevXYZ[0]/prevXYZ[2];
        float lnYX = log(ratioYX+1.718)*1.2;
        float lnZX = log(ratioZX+1.718)*2;
        xyz[1] /= lnYX;
        xyz[2] /= lnZX;
        
#if DIS
//        printf("%f %f %f %f\n", ratioYX, ratioZX, lnYX, lnZX);
        printf("%f %f %f\n", xyz[0], xyz[1], xyz[2]);
#endif
    }
    Spectrum c = RGBFromXYZ(xyz);
    return c*15;
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
