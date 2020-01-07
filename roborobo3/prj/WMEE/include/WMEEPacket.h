/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */

#ifndef WMEEPACKET_H
#define WMEEPACKET_H

#include "Utilities/Packet.h"

#include "WMEE/include/autoencoder.h"
#include "WMEE/include/mlp.h"


struct WMEEPacket : public Packet
{
    int regret = 0;

    std::shared_ptr<AE> visual_nn = nullptr;

    WMEEPacket() : Packet()
    {
    }

    void display()
    {
        Packet::display();
        std::cout << "\tregret       = " << regret << "\n";
    }

};

#endif
