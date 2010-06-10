#include "stdio.h"
#include "stdlib.h"

int main(int argc, char* argv[])
{
    int d = atoi(argv[1]);
    FILE* f = fopen("test.pbrt","w");
    
    fprintf(f, "LookAt 1.0 1.0 5   1.0 1.0 1.0    0 1 0\n");
    fprintf(f, "Camera \"perspective\" \"float fov\" [28]\n");
    fprintf(f, "Film \"image\" \"integer xresolution\" [400] \"integer yresolution\" [400]\"string filename\" \"fire-%d.exr\"\n", d);
    fprintf(f, "Sampler \"bestcandidate\" \"integer pixelsamples\" [4]\n");
    fprintf(f, "PixelFilter \"triangle\"\n");
    fprintf(f, "VolumeIntegrator \"emission\" \"float stepsize\" [.025]\n");
    fprintf(f, "WorldBegin\n");
    fprintf(f, "Volume \"firevolumegrid\"\n");
    fprintf(f, "\"point p0\" [ 0.010000 0.010000 0.010000 ] \"point p1\" [ 1.990000 1.990000 0.390000 ]\n");
    fprintf(f, "\"string simFile\" \"sliceAnimation%d.bin\"\n", d);
    fprintf(f, "\"color sigma_a\" [1 1 1] \"color sigma_s\" [0 0 0]\n");
    fprintf(f, "Material \"matte\" \"color Kd\" [.57 .57 .6]\n");
    fprintf(f, "Translate 0 -1 0\n");
    fprintf(f, "Shape \"trianglemesh\" \"integer indices\" [0 1 2 2 3 0]\n");
    fprintf(f, "\"point P\" [ -5 0 -5  5 0 -5  5 0 5  -5 0 5]\n");
    fprintf(f, "Shape \"trianglemesh\" \"integer indices\" [0 1 2 2 3 0]\n");
    fprintf(f, "\"point P\" [ -5 0 -1.7  5 0 -1.7   5 10 -1.7  -5 10 -1.7 ]\n");
    fprintf(f, "WorldEnd\n");

    fclose(f);
    return 0;
}
