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