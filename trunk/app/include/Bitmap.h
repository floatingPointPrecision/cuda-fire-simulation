/*******************************************************************************
Copyright (c) 2010, Steve Lesser
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2)Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3) The name of contributors may not be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL STEVE LESSER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/**
@file Bitmap.h
@note This file based on the code found at http://url3.tk/?p=bitmap
*/

#pragma once

#include <windows.h>
#include <string>
#include <vector_types.h>

typedef struct tagBMPFileHeader {
  DWORD bfSize; // bytes of entire file
  WORD bfReserved1; // nothing
  WORD bfReserved2; // nothing
  DWORD bfOffBits; // address of bitmap data in file
} BMPFileHeader; 

typedef struct tagBMPInfoHeader { 
  DWORD biSize; // Size of this BMPInfoHeader
  int biWidth;
  int biHeight; 
  WORD biPlanes; // 1 plane, so this equals 1
  WORD biBitCount; // 24 bits/pixel
  DWORD biCompression; // 0 = BMP
  DWORD biSizeImage; // size of actual bitmap image
  int biXPelsPerMeter; // Horizontal resolution
  int biYPelsPerMeter; // Vertical resolution
  DWORD biClrUsed; // 0
  DWORD biClrImportant; // 0 important colors (very old)
} BMPInfoHeader; 

/**
*  Bitmap class, used for writing data out to disk as a bitmap image. Will not write out to disk until flush is called.
*/
class BitmapWriter
{    
  // public methods
public:
  /**
  * constructor. Initializes the Projection object
  */
  BitmapWriter(int width, int height);
  /**
  * A destructor.
  */
  ~BitmapWriter(){free(m_data);};

  /**
  * sets the internal data as float4 where float.x = R, float.y = G, and float.z = B (no alpha)
  */
  //void setFloat4Data(float4* data);

  /**
  * sets the internal data at pixel position x, y equal to value
  * @param x x value in the image to write to
  * @param y y value in the image to write to
  * @param red red component of the pixel
  * @param green green component of the pixel
  * @param blue blue component of the pixel
  */
  void setValue(int x, int y, char red, char green, char blue);

  /**
  * flushes the headers and data out to file named fileName
  */
  void flush(const char* fileName);

  // protected members
protected:
  int m_width; ///< width of image
  int m_height; ///< height of image
  tagBMPFileHeader m_fileHeader; ///< file header information
  tagBMPInfoHeader m_fileInfoHeader; ///< file image header information
  char* m_data; ///< raw data to write out to disk
};


