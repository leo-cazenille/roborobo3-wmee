#if defined PRJ_WMEE || !defined MODULAR

#include "Config/WMEEConfigurationLoader.h"
#include "WMEE/include/WMEEWorldObserver.h"
#include "WMEE/include/WMEEAgentObserver.h"
#include "WMEE/include/WMEEController.h"
#include "WorldModels/RobotWorldModel.h"

WMEEConfigurationLoader::WMEEConfigurationLoader()
{
}

WMEEConfigurationLoader::~WMEEConfigurationLoader()
{
	//nothing to do
}

WorldObserver* WMEEConfigurationLoader::make_WorldObserver(World* wm)
{
	return new WMEEWorldObserver(wm);
}

RobotWorldModel* WMEEConfigurationLoader::make_RobotWorldModel()
{
	return new RobotWorldModel();
}

AgentObserver* WMEEConfigurationLoader::make_AgentObserver(RobotWorldModel* wm)
{
	return new WMEEAgentObserver(wm);
}

Controller* WMEEConfigurationLoader::make_Controller(RobotWorldModel* wm)
{
	return new WMEEController(wm);
}

#endif
