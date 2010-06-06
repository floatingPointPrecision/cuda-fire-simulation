#include "stdio.h"
#include "DensityTemperatureVolume.h"

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        printf("Please specify the file name");
        return -1;
    }
    int sizeX = 100, sizeY = 100, sizeZ = 40;
    DensityTemperatureVolume* data = new DensityTemperatureVolume();
    data->load(argv[1]);
    FILE* densityFile = fopen("density_test.pbrt", "w");
    fprintf(densityFile, "Volume \"volumegrid\" \"integer nx\" %d \"integer ny\" %d \"integer nz\" %d\n\"point p0\" [ 0.010000 0.010000 0.010000 ] \"point p1\" [ 1.990000 1.990000 0.790000 ]\n \"float density\" [\n", sizeX, sizeY, sizeZ);
    float scaleX = (float)(data->getSizeInX()-1)/sizeX;
    float scaleY = (float)(data->getSizeInY()-1)/sizeY;
    float scaleZ = (float)(data->getSizeInZ()-1)/sizeZ;
    for (int k = 0; k < sizeZ; k++)
    {
        for (int j = 0; j < sizeY; j++)
        {
            for (int i = 0; i < sizeX; i++)
            {
                fprintf(densityFile, "%.3f ", data->getDensityAt(i*scaleX, j*scaleY, k*scaleZ));
            }
            fprintf(densityFile, "\n");
        }
        fprintf(densityFile, "\n");
    }
    fprintf(densityFile, "]\n");
    fclose(densityFile);
}
