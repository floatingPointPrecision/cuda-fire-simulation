import numpy
from math import *
from numpy import *

maxTemperature = 1800.0

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
    #print 'wavelength:',waveLength
    #print 'temperature:',temperature
    #print 'C2:',C2
    #print 'exponent',exponentValue
    denominator = pow(waveLength,5) * (exp(exponentValue) - 1)
    #print 'wavelength to the 5th',pow(waveLength,5)
    #print 'other denominator part',(exp(exponentValue) - 1)
    radiance = numerator / denominator
    #print 'radiance:',radiance
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
print 'raw XYZ:',xyz
#normalizeList(xyz,3)

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

def XYZtoRGB(x,y,z):
    xyz = [x,y,z]
    normalizeList(xyz,3)
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]
    r = 2.5623 * x + (-1.1661) * y + (-0.3962) * z
    g = (-1.0215) * x + 1.9778 * y + 0.0437 * z
    b = 0.0752 * x + (-0.2562) * y + 1.1810 * z
    return [r,g,b]

def generateRGBfromTemp(temp):
    xyz = [0, 0, 0]
    sampleRange = lambdaValues[-1] - lambdaValues[0]
    sampleIncrement = sampleRange / numSamples
    for sample in range(numSamples):
        wavelength = lambdaValues[sample]
        radiance = getRadiance(wavelength, temp)
        xyz[0] += cieX[sample]*radiance
        xyz[1] += cieY[sample]*radiance
        xyz[2] += cieZ[sample]*radiance
    return XYZtoRGB(xyz[0],xyz[1],xyz[2])

def generateRGBfromTempWithWhiteNormalization(temp):
    xyz = [0, 0, 0]
    sampleRange = lambdaValues[-1] - lambdaValues[0]
    sampleIncrement = sampleRange / numSamples
    for sample in range(numSamples):
        wavelength = lambdaValues[sample]
        radiance = getRadiance(wavelength, temp)
        xyz[0] += cieX[sample]*radiance
        xyz[1] += cieY[sample]*radiance
        xyz[2] += cieZ[sample]*radiance
    xyzAsMatrix = matrix('0.0; 0; 0')
    xyzAsMatrix[0]=xyz[0]
    xyzAsMatrix[1]=xyz[1]
    xyzAsMatrix[2]=xyz[2]
    normalizeList(xyz,3)
    finalXYZ = finalTransformMatrix * xyzAsMatrix
    x = int(finalXYZ[0])
    y = int(finalXYZ[1])
    z = int(finalXYZ[2])
    return XYZtoRGB(x,y,z)
print ''
print 'RGB values'
for t in range(50):
    temp = (t+1) * 50
    rgb = generateRGBfromTemp(temp)
    rgb[0]=min(max(rgb[0],0),1)
    rgb[1]=min(max(rgb[1],0),1)
    rgb[2]=min(max(rgb[2],0),1)
    print 'rgb value',temp,rgb[0]*255,rgb[1]*255,rgb[2]*255
    
    




