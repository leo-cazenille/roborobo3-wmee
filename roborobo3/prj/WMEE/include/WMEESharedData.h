/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */



#ifndef WMEESHAREDDATA_H
#define WMEESHAREDDATA_H

#include "TemplateEE/include/TemplateEESharedData.h"

class WMEESharedData : TemplateEESharedData {
    
    // cf. super class for many parameter values.
    // Add here parameters that are specific to current implementation.

public:
    static int nbObjectsType1;
    static int nbObjectsType2;
    static int foragingTask;
    static int fitnessFunction;
    static int regretValue;

    static int gControllerType; // controller type (0: MLP, 1: Perceptron, 2: Elman, 4: AE+MLP)
    static int rebirthDelay;

    static int dataBaseMaxCapacity;
    static int maxStoredVisualModels;
    static double learningRate;

    static int phase2AfterIt;
    static int phase2ControllerType;
    static int aeHDim;
    static int aeZDim;
}; 

#endif
