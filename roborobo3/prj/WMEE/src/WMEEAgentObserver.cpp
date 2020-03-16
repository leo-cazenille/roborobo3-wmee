/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */

#include "WMEE/include/WMEEAgentObserver.h"
#include "WMEE/include/WMEESharedData.h"
#include "WMEE/include/WMEEController.h"
#include "WorldModels/RobotWorldModel.h"
#include "World/PhysicalObject.h"
#include "RoboroboMain/roborobo.h"
#include "World/World.h"


WMEEAgentObserver::WMEEAgentObserver( RobotWorldModel *wm ) : TemplateEEAgentObserver ( wm )
{
    // superclass constructor called before
}

WMEEAgentObserver::~WMEEAgentObserver()
{
    // superclass destructor called before
}

/*
 * Manage foraging of energy items.
 * If walked on, the item disappear (callback to item object) and agent's fitness function is updated.
 * Assume that *only* energy point's footprint can be walked upon. That means that for this code to correctly run you should avoid placing any objects detectable through the robot's ground sensor.
 *
 */
void WMEEAgentObserver::stepPre()
{
    // * update fitness (if needed)
    if ( _wm->isAlive() && PhysicalObject::isInstanceOf(_wm->getGroundSensorValue()) )
    {
        //if ( WMEESharedData::foragingTask != 0 && WMEESharedData::foragingTask != 1 && WMEESharedData::foragingTask != 2 )
        if ( WMEESharedData::foragingTask > 3)
        {
            std::cerr << "[ERROR] gForagingTask value is unknown. Exiting.\n";
            exit (-1);
        }

        if ( WMEESharedData::foragingTask == 0 )
        {
            _wm->_fitnessValue = _wm->_fitnessValue + 1;
        }

        int targetIndex = _wm->getGroundSensorValue() - gPhysicalObjectIndexStartOffset;
        int threshold = WMEESharedData::nbObjectsType1;
        if ( gPhysicalObjects[targetIndex]->getId() < threshold )
        {
            WMEEController *ctl = dynamic_cast<WMEEController*>(getController());
            if ( WMEESharedData::foragingTask == 1 ) {
                _wm->_fitnessValue = _wm->_fitnessValue + 1;
                ctl->nbForagedItemType0++;
            } else if ( WMEESharedData::foragingTask == 2 ) {
                if ( ctl->nbForagedItemType1 - ctl->nbForagedItemType0 >= 0 ) {
                    _wm->_fitnessValue += 1;
                    ctl->nbForagedItemType0++;
                } else {
                    _wm->_fitnessValue = 0;
                }
            } else if ( WMEESharedData::foragingTask == 3 ) {
                if ( ctl->nbForagedItemType1 - ctl->nbForagedItemType0 >= 0 ) {
                    _wm->_fitnessValue += 1;
                    ctl->nbForagedItemType0++;
                } else {
                    _wm->_fitnessValue -= 1;
                }
            } else {
                ctl->nbForagedItemType0++;
            }
        }
        else
        {
            WMEEController *ctl = dynamic_cast<WMEEController*>(getController());
            if ( WMEESharedData::foragingTask == 1 ) {
                _wm->_fitnessValue = _wm->_fitnessValue - 1;
                ctl->nbForagedItemType1++;
            } else if ( WMEESharedData::foragingTask == 2 ) {
                if ( ctl->nbForagedItemType0 - ctl->nbForagedItemType1 >= 0 ) {
                    _wm->_fitnessValue += 1;
                    ctl->nbForagedItemType1++;
                } else {
                    _wm->_fitnessValue = 0;
                }
            } else if ( WMEESharedData::foragingTask == 3 ) {
                if ( ctl->nbForagedItemType0 - ctl->nbForagedItemType1 >= 0 ) {
                    _wm->_fitnessValue += 1;
                    ctl->nbForagedItemType1++;
                } else {
                    _wm->_fitnessValue -= 1;
                }
            } else {
                ctl->nbForagedItemType1++;
            }
        }

    }

    TemplateEEAgentObserver::stepPre();
}



