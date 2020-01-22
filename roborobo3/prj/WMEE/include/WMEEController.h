/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */

#ifndef WMEECONTROLLER_H
#define WMEECONTROLLER_H

#include <memory>
#include <deque>
#include "TemplateEE/include/TemplateEEController.h"
#include "WMEE/include/WMEEPacket.h"

#include "WMEE/include/autoencoder.h"
#include "WMEE/include/lstm.h"
#include "WMEE/include/mlp.h"
#include <torch/torch.h>

using namespace Neural;

class RobotWorldModel;

class WMEEController : public TemplateEEController
{
protected:
    int _rebirthDelay;

    std::shared_ptr<AE> _visual_nn = nullptr;
    //std::vector<decltype(_visual_nn)> _vec_visual_nn;
    std::map< std::pair<int,int>, decltype(_visual_nn)> _vec_visual_nn;

    std::shared_ptr<MDNLSTM> _memory_nn = nullptr;
    std::map< std::pair<int,int>, decltype(_memory_nn)> _vec_memory_nn;

    torch::Tensor _data;
    size_t _data_idx = 0;

    std::vector<double> _last_outputs;
    std::deque<std::vector<double>> _last_data_mem;
    //torch::Tensor _last_data_mem;
    //size_t _last_data_mem_idx = 0;

    torch::Tensor _data_mem;
    size_t _data_mem_idx = 0;

//    std::unique_ptr<torch::optim::Adam> _optimizer = nullptr;

private:
    std::map< std::pair<int,int>, int > _regretValueList;
    
public:
    
    WMEEController(RobotWorldModel *wm);
    ~WMEEController();
    
    double getFitness() override;
    
    int nbForagedItemType0;
    int nbForagedItemType1;
    
    int* lastSeenObjectIdPerSensorList = NULL;
    //int lastSeenObjectIdOnFloorSensor = 0;
    
    int regret;
    
protected:
    
    void initController() override;
    void initEvolution() override;
    
    void step() override;
    void stepController() override;
    void stepEvolution() override;
    
    void performSelection() override;
    void performVariation() override;
    
    void broadcastGenome() override;
    
    void resetFitness() override;
    void updateFitness() override;
    
    void logCurrentState() override;
    
    bool sendGenome( TemplateEEController* __targetRobotController ) override;
    bool receiveGenome( Packet* p ) override;
    
    void selectNaiveMO();
    
    void createNN() override;
    double testAE(std::shared_ptr<AE> ae);
    void trainAE();

    double testMM(std::shared_ptr<MDNLSTM> lstm);
    void trainMM();
};


#endif

