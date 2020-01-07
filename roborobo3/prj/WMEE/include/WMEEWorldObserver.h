/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */

#ifndef WMEEWORLDOBSERVER_H
#define WMEEWORLDOBSERVER_H

#include "TemplateEE/include/TemplateEEWorldObserver.h"

class World;

class WMEEWorldObserver : public TemplateEEWorldObserver
{
public:
    WMEEWorldObserver(World *world);
    ~WMEEWorldObserver();
    
    void initPre();
    void initPost();
    
    void stepPre();
    void stepPost();

protected:    
    virtual void monitorPopulation( bool localVerbose = true );
};

#endif
