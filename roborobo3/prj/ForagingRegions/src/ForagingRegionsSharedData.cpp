/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */



#include "ForagingRegions/include/ForagingRegionsSharedData.h"

// cf. super class for many parameter values initialization.
// Add here initialization for parameters that are specific to current implementation.
//
// Quick help:
//  if parameter is already defined in TemplateEEShareData, then use TemplateEEShareData::parametername
//  to define a new parameter, do it in ForagingRegionsSharedData.h, initialize it in ForagingRegionsSharedData.cpp, then use ForagingRegionsSharedData::parametername
//

int ForagingRegionsSharedData::nbObjectsOnLeft = 0;
int ForagingRegionsSharedData::nbObjectsOnRight = 0;