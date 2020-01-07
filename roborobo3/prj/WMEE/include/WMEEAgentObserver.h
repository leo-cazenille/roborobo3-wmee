/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */

#ifndef WMEEAGENTOBSERVER_H
#define WMEEAGENTOBSERVER_H

#include "TemplateEE/include/TemplateEEAgentObserver.h"

class RobotWorldModel;

class WMEEAgentObserver : public TemplateEEAgentObserver
{
	public:
		WMEEAgentObserver(RobotWorldModel *wm);
		~WMEEAgentObserver();
    
        void stepPre() override;
};

#endif

