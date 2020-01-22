/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */

#ifndef WMEEPACKET_H
#define WMEEPACKET_H

#include "Utilities/Packet.h"

#include "WMEE/include/autoencoder.h"
#include "WMEE/include/lstm.h"
#include "WMEE/include/mlp.h"

#include <torch/torch.h>


struct WMEEPacket : public Packet
{
    int regret = 0;

    std::shared_ptr<AE> visual_nn = nullptr;
    std::shared_ptr<MDNLSTM> memory_nn = nullptr;

//    std::shared_ptr<torch::optim::Adam> optim_visual_nn = nullptr;

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
