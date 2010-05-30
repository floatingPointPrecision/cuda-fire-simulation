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
@file Bitmap.cpp
@brief Implements the Bitmap class which can write out values to a bitmap
*/

#include "Bitmap.h"

BitmapWriter::BitmapWriter(int width, int height)
: m_width(width),m_height(height),m_data(0)
{
  int imagebytes=width*height*3;
  m_fileHeader.bfSize = 2+sizeof(BMPFileHeader)+sizeof(BMPInfoHeader)+imagebytes;
  m_fileHeader.bfReserved1 = 0; 
  m_fileHeader.bfReserved2 = 0;
  m_fileHeader.bfOffBits = 2+sizeof(BMPFileHeader)+sizeof(BMPInfoHeader);
  m_fileInfoHeader.biSize = sizeof(BMPInfoHeader);
  m_fileInfoHeader.biWidth = width;
  m_fileInfoHeader.biHeight = height;
  m_fileInfoHeader.biPlanes = 1;
  m_fileInfoHeader.biBitCount = 24; // 24 bits/pixel
  m_fileInfoHeader.biCompression = 0; // Zero is the defaut Bitmap
  m_fileInfoHeader.biSizeImage = imagebytes;
  m_fileInfoHeader.biXPelsPerMeter = 2835; // 72 pixels/inch = 2834.64567 pixels per meter
  m_fileInfoHeader.biYPelsPerMeter = 2835;
  m_fileInfoHeader.biClrUsed = 0;
  m_fileInfoHeader.biClrImportant = 0;
  m_data =(char*) malloc(imagebytes);


  // Now to write the file
  // Open the file in "write binary" mode

}

void BitmapWriter::flush(const char* fileName)
{
  FILE *f = 0;
  fopen_s(&f, fileName,"wb");
  if (f == 0)
    return;
  char magic[2]={'B','M'};
  fwrite(magic, 2, 1, f);
  fwrite((void*)&m_fileHeader, sizeof(BMPFileHeader), 1, f);
  fwrite((void*)&m_fileInfoHeader, sizeof(BMPInfoHeader), 1, f);
  fwrite(m_data, m_fileInfoHeader.biSizeImage, 1, f);
  fclose(f);
}

void BitmapWriter::setValue(int x, int y, char red, char green, char blue)
{
  if (x < 0 || x >= m_width || y < 0 || y >= m_height)
    return;
  m_data[((y*m_width)+x)*3] = red;
  m_data[((y*m_width)+x)*3+1] = green;
  m_data[((y*m_width)+x)*3+2] = blue;
}

//void BitmapWriter::setFloat4Data(float4* data)
//{
//  int x,y;
//  for(x=0;x<width;++x)
//  {
//    for(y=0;y<height;++y)
//    {
//      // R component
//      m_data[((y*width)+x)*3] = char(data[y*width+x].x);
//      // G component
//      m_data[((y*width)+x)*3+1] = char(data[y*width+x].y);
//      // B component
//      m_data[((y*width)+x)*3+2] = char(data[y*width+x].z);
//    }
//  }
//}