/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */



#include "WMEE/include/WMEESharedData.h"

// cf. super class for many parameter values initialization.
// Add here initialization for parameters that are specific to current implementation.
//
// Quick help:
//  if parameter is already defined in TemplateEEShareData, then use TemplateEEShareData::parametername
//  to define a new parameter, do it in WMEESharedData.h, initialize it in WMEESharedData.cpp, then use WMEESharedData::parametername
//

int WMEESharedData::nbObjectsType1 = 0;
int WMEESharedData::nbObjectsType2 = 0;
int WMEESharedData::foragingTask = 0;
int WMEESharedData::fitnessFunction = 0;
int WMEESharedData::regretValue = 0;

int WMEESharedData::gControllerType = 0;
int WMEESharedData::rebirthDelay = 0;

int WMEESharedData::dataBaseMaxCapacity = 0;
int WMEESharedData::dataMemNbSequences = 0;
int WMEESharedData::maxStoredVisualModels = 0;
int WMEESharedData::maxStoredMemoryModels = 0;
double WMEESharedData::learningRate = 1e-3;
double WMEESharedData::learningRateMDNLSTM = 1e-3;

int WMEESharedData::phase2AfterIt = 0;
int WMEESharedData::phase2ControllerType = 0;
int WMEESharedData::phase3AfterIt = 0;
int WMEESharedData::phase3ControllerType = 0;
int WMEESharedData::aeHDim = 0;
int WMEESharedData::aeZDim = 0;
int WMEESharedData::mdnlstmHDim = 0;
//int WMEESharedData::mdnlstmZDim = 0;
int WMEESharedData::mdnlstmNbSamples = 0;
int WMEESharedData::mdnlstmHiddenDim = 0;
int WMEESharedData::mdnlstmNbLayers = 0;
double WMEESharedData::mdnlstmTemperature = 0;
int WMEESharedData::stopTrainingAfterIt = 0;

