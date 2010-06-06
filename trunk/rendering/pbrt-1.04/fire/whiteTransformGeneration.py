import numpy
from math import *
from numpy import *

maxTemperature = 1700.0

# XYZ to LMS transform matrix from http://en.wikipedia.org/wiki/LMS_color_space
XYZtoLMS = matrix('0.8951 0.2664 -0.1614; -0.7502 1.7135 0.0367; 0.0389 -0.0685 1.0296')
XYZtoLMSinv = XYZtoLMS.getI()
print 'XYZ to LMS transform matrix'
print XYZtoLMS
print ''
print 'LMS to XYZ inverse transform matrix'
print XYZtoLMSinv

# CIE color matching functions for discrete wavelengths
numSamples = 10
lambdaValues = [360, 410, 460, 510, 560, 610, 660, 710, 760, 810]
nanoToMeters = 10**(-9)
for i in range(numSamples):
    lambdaValues[i] = lambdaValues[i] * nanoToMeters   
cieX = [0.0001299, 0.04351, 0.2908, 0.0093, 0.5945, 1.0026, 0.1649, 0.005790346, 0.000166151, 0.00000508587]
cieY = [0.000003917, 0.00121, 0.06, 0.503, 0.995, 0.503, 0.061, 0.002091, 0.00006, 0.0000018366]
cieZ = [0.0006061, 0.2074, 1.6692, 0.1582, 0.0039, 0.00034, 0, 0, 0, 0]
def normalizeList(v, length):
    listSum = 0
    for i in range(length):
        listSum += v[i]
    for i in range(length):
        v[i] /= listSum
normalizeList(cieX,numSamples)
normalizeList(cieY,numSamples)
normalizeList(cieZ,numSamples)

# get radiance from wavelength and temperature as determined by
# http://physbam.stanford.edu/~fedkiw/papers/stanford2002-02.pdf
def getRadiance(waveLength, temperature):
    C1 = 3.7418 * (10**(-16))
    C2 = 1.4388 * (10**(-2))
    numerator = 2 * C1
    exponentValue = C2 / (waveLength*temperature)
    denominator = pow(wavelength,5) * (exp(exponentValue) - 1)
    radiance = numerator / denominator
    return radiance

# generate xyz coordinates for white point
xyz = [0, 0, 0]
sampleRange = lambdaValues[-1] - lambdaValues[0]
sampleIncrement = sampleRange / numSamples
for sample in range(numSamples):
    wavelength = lambdaValues[sample]
    radiance = getRadiance(wavelength, maxTemperature)
    xyz[0] += cieX[sample]*radiance
    xyz[1] += cieY[sample]*radiance
    xyz[2] += cieZ[sample]*radiance
normalizeList(xyz,3)

# convert white point xyz coordinates to LMS coordinates
xyzAsMatrix = matrix('0.0; 0; 0')
xyzAsMatrix[0]=xyz[0]
xyzAsMatrix[1]=xyz[1]
xyzAsMatrix[2]=xyz[2]
print ''
print 'xyz white point',xyzAsMatrix
lmsWhitePoint = XYZtoLMS * xyzAsMatrix
print ''
print 'lms white point',lmsWhitePoint

# generate white point normalization LMS matrix
whitePointDiagMatrix = matrix('0.0 0 0; 0 0 0; 0 0 0')
whitePointDiagMatrix.put(0,1/lmsWhitePoint[0])
whitePointDiagMatrix.put(4,1/lmsWhitePoint[1])
whitePointDiagMatrix.put(8,1/lmsWhitePoint[2])
print ''
print 'diagonal white point matrix:'
print whitePointDiagMatrix

# generate final transformation matrix
finalTransformMatrix = XYZtoLMSinv * whitePointDiagMatrix * XYZtoLMS
print ''
print 'final transform matrix:'
print finalTransformMatrix




