/*
 * WMEEConfigurationLoader.h
 */

#ifndef WMEECONFIGURATIONLOADER_H
#define WMEECONFIGURATIONLOADER_H

#include "Config/ConfigurationLoader.h"

class WMEEConfigurationLoader : public ConfigurationLoader
{
	private:

	public:
		WMEEConfigurationLoader();
		~WMEEConfigurationLoader();

		WorldObserver *make_WorldObserver(World* wm) ;
		RobotWorldModel *make_RobotWorldModel();
		AgentObserver *make_AgentObserver(RobotWorldModel* wm) ;
		Controller *make_Controller(RobotWorldModel* wm) ;
};



#endif
