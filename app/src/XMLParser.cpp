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
@file XMLParser.cpp
*/

#include "XMLParser.h"

XMLParser::XMLParser(const char* fileName)
: m_root(0)
{
  strcpy_s(m_fileName, 128, fileName);
  m_document = TiXmlDocument(fileName);
  m_document.LoadFile();
  m_root = m_document.RootElement();
}
    
XMLParser::~XMLParser()
{
}

bool XMLParser::resetRoot()
{
  m_root = m_document.RootElement();
  return true;
}

bool XMLParser::setNewRoot(const char* rootName)
{
  if (m_root == 0)
    return false;
  TiXmlNode* temp = m_root;
  m_root = m_root->FirstChild(rootName);
  if (m_root==0)
  {
    m_root = temp;
    return false;
  }
  else
  {
    return true;
  }
}

bool XMLParser::getInt(const char* attributeName, int* result)
{
  if (m_root == 0)
    return false;
  TiXmlElement* currentAttribute = m_root->FirstChild(attributeName)->ToElement();
  if (currentAttribute)
  {
    *result = (int) atoi(currentAttribute->GetText());
    return true;
  }
  return false;
}

bool XMLParser::getFloat(const char* attributeName, float* result)
{
  if (m_root == 0)
    return false;
  TiXmlElement* currentAttribute = m_root->FirstChild(attributeName)->ToElement();
  if (currentAttribute)
  {
    *result = (float) atof(currentAttribute->GetText());
    return true;
  }
  return false;
}

 bool XMLParser::getFloat2(const char* attributeName, float result[2])
 {
   if (m_root == 0)
    return false;
  TiXmlElement* currentAttribute = m_root->FirstChild(attributeName)->ToElement();
  if (currentAttribute)
  {
    result[0] = (float)atof(currentAttribute->FirstChild("x")->ToElement()->GetText());
    result[1] = (float)atof(currentAttribute->FirstChild("y")->ToElement()->GetText());
    return true;
  }
  return false;
 }

 bool XMLParser::getFloat3(const char* attributeName, float result[3])
 {
   if (m_root == 0)
    return false;
  TiXmlElement* currentAttribute = m_root->FirstChild(attributeName)->ToElement();
  if (currentAttribute)
  {
    result[0] = (float)atof(currentAttribute->FirstChild("x")->ToElement()->GetText());
    result[1] = (float)atof(currentAttribute->FirstChild("y")->ToElement()->GetText());
    result[2] = (float)atof(currentAttribute->FirstChild("z")->ToElement()->GetText());
    return true;
  }
  return false;
 }