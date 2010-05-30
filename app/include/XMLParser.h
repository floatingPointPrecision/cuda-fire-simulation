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
@file XMLParser.h
*/

#pragma once

#include "tinyxml/tinyxml.h"

/**
  * simple parser for reading in XML files and nodes
  */
class XMLParser
{
public:
  /**
  * default constructor, does not load any files
  */
  XMLParser() {}
  /**
  * basic constructor, starts at the root node of the file
  */
  XMLParser(const char* fileName);
  /**
  * destructor
  */
  ~XMLParser();
  /**
  * sets the root node to the root of the document
  * @return true if successful, false otherwise
  */
  bool resetRoot();
  /**
  * sets the root node which is the parent point for future searches
  * @param rootName name of the new root node. Must be a child of the current root
  * @return true if successful, false otherwise
  */
  bool setNewRoot(const char* rootName);
  /**
  * sets the result int value to the value found in the attribute name node
  * @param attributeName name of the node to find the int. Node must be a child of the current root node 
  * @param result pointer to the location to store the attribute's value
  * @return true if successful, false otherwise
  */
  bool getInt(const char* attributeName, int* result);
  /**
  * sets the result unsigned int value to the value found in the attribute name node
  * @param attributeName name of the node to find the unsigned int. Node must be a child of the current root node 
  * @param result pointer to the location to store the attribute's value
  * @return true if successful, false otherwise
  */
  bool getUnsignedInt(const char* attributeName, unsigned int* result);
  /**
  * sets the result float value to the value found in the attribute name node
  * @param attributeName name of the node to find the float. Node must be a child of the current root node
  * @param result pointer to the location to store the attribute's value
  * @return true if successful, false otherwise
  */
  bool getFloat(const char* attributeName, float* result);
  /**
  * sets the result 2 float values to the values found in the node's x and y children nodes.
  * @param attributeName name of the node to find the floats. Node must be a child of the current root node
  * @param result array specifying the location to store the attribute's value
  * @return true if successful, false otherwise
  * @note Given the xml data:
  * \code 
  * <xLowerBound>
  *   <x>-1</x>
  *   <y>-1</y>
  * </xLowerBound> \endcode
  * with code:
  * \code 
  * float floatArray[2]; 
  * getFloat2("xLowerBound",floatArray)
  * \endcode
  * floatArray will become populated as {-1,-1}
  */
  bool getFloat2(const char* attributeName, float result[2]);
  /**
  * sets the result 3 float values to the values found in the node's x y and z children nodes.
  * @param attributeName name of the node to find the floats. Node must be a child of the current root node
  * @param result array specifying the location to store the attribute's value
  * @return true if successful, false otherwise
  * @note Given the xml data:
  * \code 
  * <lowerLeftCorner>
  *   <x>-1</x>
  *   <y>-1</y>
  *   <z>-1</z>
  * </lowerLeftCorner> \endcode
  * with code:
  * \code 
  * float floatArray[3]; 
  * getFloat3("lowerLeftCorner",floatArray)
  * \endcode
  * floatArray will become populated as {-1,-1,-1}
  */
  bool getFloat3(const char* attributeName, float result[3]);

private:
  bool m_error;
  char m_fileName[128];
  TiXmlDocument m_document;
  TiXmlNode* m_root;
};