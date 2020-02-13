/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */

#include "WMEE/include/WMEEController.h"
#include "WMEE/include/WMEESharedData.h"
#include "World/World.h"
#include "RoboroboMain/roborobo.h"
#include "WorldModels/RobotWorldModel.h"
#include <algorithm>
#include <math.h>
#include "neuralnetworks/MLP.h"
#include "neuralnetworks/Perceptron.h"
#include "neuralnetworks/Elman.h"
#include "neuralnetworks/ESNEigen.h"



using namespace Neural;

WMEEController::WMEEController( RobotWorldModel *wm ) : TemplateEEController( wm )
{
    //_data.resize(WMEESharedData::dataBaseMaxCapacity);
    _data = torch::zeros({WMEESharedData::dataBaseMaxCapacity, _nbInputs});
    _data_mem = torch::zeros({WMEESharedData::dataBaseMaxCapacity, WMEESharedData::dataMemNbSequences, WMEESharedData::aeZDim + _nbOutputs}); // XXX
    //_last_data_mem = torch::zeros({WMEESharedData::dataMemNbSequences, WMEESharedData::aeZDim + _nbOutputs}); // XXX

    // superclass constructor called before this baseclass constructor.
    resetFitness(); // superconstructor calls parent method.
    
    lastSeenObjectIdPerSensorList = new int [_wm->_cameraSensorsNb];
    for ( int i = 0 ; i < _wm->_cameraSensorsNb ; i++ )
        lastSeenObjectIdPerSensorList[i] = -1;
    
    //lastSeenObjectIdOnFloorSensor = -1;
}

WMEEController::~WMEEController()
{
    // superclass destructor automatically called after this baseclass destructor.
    
    delete [] lastSeenObjectIdPerSensorList;

    _visual_nn.reset();
    _memory_nn.reset();
}


void WMEEController::step() // handles control decision and evolution (but: actual movement is done in roborobo's main loop)
{
    _iteration++;

    // Update controller type, if necessary
    if (WMEESharedData::phase2AfterIt == _iteration) {
        std::cout << "PHASE CHANGE 1->2 !!" << std::endl;
        WMEESharedData::gControllerType = WMEESharedData::phase2ControllerType;
        createNN();
    }
    if (WMEESharedData::phase3AfterIt == _iteration) {
        std::cout << "PHASE CHANGE 2->3 !!" << std::endl;
        WMEESharedData::gControllerType = WMEESharedData::phase3ControllerType;
        createNN();
    }

    // * step evolution
    stepEvolution();
    
    // * step controller
    if ( _wm->isAlive() ) {
        stepController();
        updateFitness();
    } else {
        _wm->_desiredTranslationalValue = 0.0;
        _wm->_desiredRotationalVelocity = 0.0;
    }
    
    // * updating listening state
    if ( _wm->isAlive() == false ) {
        assert ( _notListeningDelay >= -1 ); // -1 means infinity
        ++_rebirthDelay;
        if (_rebirthDelay >= WMEESharedData::rebirthDelay) {
            _wm->setAlive(true);
            _rebirthDelay = 0;
        }
        
        if ( _notListeningDelay > 0 ) {
            _notListeningDelay--;
            
            if ( _notListeningDelay == 0 ) {
                
                _listeningDelay = TemplateEESharedData::gListeningStateDelay;
                
                if ( _listeningDelay > 0 || _listeningDelay == -1 ) {
                    _isListening = true;
                    
                    _wm->setRobotLED_colorValues(0, 255, 0); // is listening
                    
                    std::string sLog = std::string("");
                    sLog += "" + std::to_string(gWorld->getIterations()) + "," + std::to_string(_wm->getId()) + "::" + std::to_string(_birthdate) + ",status,listening\n";
                    gLogManager->write(sLog);
                    gLogManager->flush();
                } else {
                    std::string sLog = std::string("");
                    sLog += "" + std::to_string(gWorld->getIterations()) + "," + std::to_string(_wm->getId()) + "::" + std::to_string(_birthdate) + ",status,inactive\n"; // never listen again.
                    gLogManager->write(sLog);
                    gLogManager->flush();
                }
            }
        } else {
            if ( _notListeningDelay != -1 && _listeningDelay > 0 ) {
                assert ( _isListening == true );
                
                _listeningDelay--;
                
                if ( _listeningDelay == 0 ) {
                    _isListening = false;
                    // Logging: robot is dead
                    std::string sLog = std::string("");
                    sLog += "" + std::to_string(gWorld->getIterations()) + "," + std::to_string(_wm->getId()) + "::" + std::to_string(_birthdate) + ",status,inactive\n";
                    gLogManager->write(sLog);
                    gLogManager->flush();

                    _notListeningDelay = -1; // agent will not be able to be active anymore
                    _wm->setRobotLED_colorValues(0, 0, 255); // is not listening
                    
                    reset(); // destroy then create a new NN
                    
                    _wm->setAlive(false);
                }
            }
        }
    } else {
        _rebirthDelay = 0;
    }
}


void WMEEController::stepController()
{
    // ---- compute and read out ----
    
    nn->setWeights(_parameters); // set-up NN
    
    std::vector<double> inputs = getInputs(); // Build list of inputs (check properties file for extended/non-extended input values

    // Update vision model database
    auto idx = _data_idx % (size_t)WMEESharedData::dataBaseMaxCapacity;
    auto data_access = _data.accessor<float,2>();
    for(int j = 0; j < data_access.size(1); ++j) {
        if(isnormal(inputs[j]))
            data_access[idx][j] = (1.0 + inputs[j]) / 2.0; // Normalization from [-1, 1] to [0, 1]
        else
            data_access[idx][j] = 0.0;
    }
    ++_data_idx;

    std::vector<double> vec_mem;

    switch ( WMEESharedData::gControllerType )
    {
        case 4: // AE+MLP
        {
            torch::NoGradGuard no_grad;
            _visual_nn->get()->eval();

            // Compute visual model
            //torch::Tensor tensor_inputs = torch::from_blob(inputs.data(), {(long)inputs.size()}); // XXX
            auto tensor_inputs = torch::empty({(long)inputs.size()});
            auto tensor_inputs_access = tensor_inputs.accessor<float,1>();
            for(int j = 0; j < tensor_inputs_access.size(0); ++j) {
                if(isnormal(inputs[j]))
                    tensor_inputs_access[j] = (1.0 + inputs[j]) / 2.0; // Normalization from [-1, 1] to [0, 1]
                else
                    tensor_inputs_access[j] = 0.0;
            }
            auto vision_nn_output = _visual_nn->get()->forward(tensor_inputs);
            auto z_tensor = vision_nn_output.z;

            std::vector<double> z;
            size_t z_size = z_tensor.numel();
            auto p = static_cast<float*>(z_tensor.storage().data());
            for(size_t i=0; i<z_size; ++i) {
                z.push_back(p[i]);
            }
            vec_mem = z;
            //std::cout << "Z: " << z << std::endl;
            //std::vector<double> z(z_tensor.data<double>());

            nn->setInputs(z);
        }

        case 5: // MDNLSTM+AE+MLP
        {
            torch::NoGradGuard no_grad;
            _visual_nn->get()->eval();
            _memory_nn->get()->eval();

            // Compute visual model
            //torch::Tensor tensor_inputs = torch::from_blob(inputs.data(), {(long)inputs.size()}); // XXX
            auto tensor_inputs = torch::empty({(long)inputs.size()});
            auto tensor_inputs_access = tensor_inputs.accessor<float,1>();
            for(int j = 0; j < tensor_inputs_access.size(0); ++j) {
                if(isnormal(inputs[j]))
                    tensor_inputs_access[j] = (1.0 + inputs[j]) / 2.0; // Normalization from [-1, 1] to [0, 1]
                else
                    tensor_inputs_access[j] = 0.0;
            }
            auto vision_nn_output = _visual_nn->get()->forward(tensor_inputs);
            auto z_tensor = vision_nn_output.z;

            std::vector<double> z;
            int64_t const z_size = z_tensor.numel();
            auto p = static_cast<float*>(z_tensor.storage().data());
            for(int64_t i=0; i<z_size; ++i) {
                z.push_back(p[i]);
            }
            vec_mem = z;
            //std::cout << "Z: " << z << std::endl;
            //std::vector<double> z(z_tensor.data<double>());

            // Reset the hidden state of the LSTM every nb_sequences iterations
            if(_iteration % WMEESharedData::dataMemNbSequences == 0) {
                _memory_nn->get()->reset_hidden();
            }

            // Build the inputs of the controllers from latent representation from the VAE and the hidden states of the LSTM
            auto lstm_hidden = _memory_nn->get()->get_hidden();
            int64_t const lstm_hidden_size = lstm_hidden.numel();
            auto p2 = static_cast<float*>(lstm_hidden.storage().data());
            for(int64_t i=0; i<lstm_hidden_size; ++i) {
                z.push_back(p2[i]);
            }
            nn->setInputs(z);

            // Update LSTM
            auto lstm_tensor_inputs = torch::empty({1, z_size + _nbOutputs});
            auto lstm_tensor_inputs_access = lstm_tensor_inputs.accessor<float,2>();
            for(int64_t j = 0; j < z_size; ++j) {
                if(isnormal(z[j]))
                    lstm_tensor_inputs_access[0][j] = z[j];
                else
                    lstm_tensor_inputs_access[0][j] = 0.0;
            }
            for(int64_t j = 0; j < _nbOutputs; ++j) {
                if(isnormal(_last_outputs[j]))
                    lstm_tensor_inputs_access[0][j] = _last_outputs[j];
                else
                    lstm_tensor_inputs_access[0][j] = 0.0;
            }
            auto const memory_nn_output = _memory_nn->get()->forward(lstm_tensor_inputs);
        }

		default:
            nn->setInputs(inputs);
    }

    
    switch ( WMEESharedData::gControllerType )
    {
        case 3: // ESN: multiple NN updates are possible (reservoir contains recurrent connections)
		{
			static_cast<ESNEigen*>(nn)->step(static_cast<size_t>(TemplateEESharedData::gESNStepsBySimulationStep));
			break;
		}
		default:
			nn->step();
	}
    
    std::vector<double> outputs = nn->readOut();
    _last_outputs = outputs;


    if(vec_mem.size() > 0) {
        // Update last entry of memory database
        for(size_t i=0; i<outputs.size(); ++i) {
            vec_mem.push_back(outputs[i]);
        }
        _last_data_mem.push_back(vec_mem);
        if(_last_data_mem.size() >= (size_t)WMEESharedData::dataMemNbSequences)
            _last_data_mem.pop_front();

        // NOTE Really not optimized !
        // Update memory model database
        auto midx = _data_mem_idx % (size_t)WMEESharedData::dataBaseMaxCapacity;
        auto data_mem_access = _data_mem.accessor<float,3>();
        for(size_t j = 0; j < (size_t)data_mem_access.size(1); ++j) {
            for(size_t k = 0; k < (size_t)data_mem_access.size(2); ++k) {
                if(_last_data_mem.size() > j && isnormal(_last_data_mem[j][k]))
                    data_mem_access[midx][j][k] = (1.0 + _last_data_mem[j][k]) / 2.0; // Normalization from [-1, 1] to [0, 1]
                else
                    data_mem_access[midx][j][k] = 0.0;
            }
        }
        ++_data_mem_idx;
    }

    
    // std::cout << "[DEBUG] Neural Network :" << nn->toString() << " of size=" << nn->getRequiredNumberOfWeights() << std::endl;
   
    if (isnormal(outputs[0]))
        _wm->_desiredTranslationalValue = outputs[0];
    else
        _wm->_desiredTranslationalValue = 0.;
    if (isnormal(outputs[1]))
        _wm->_desiredRotationalVelocity = outputs[1];
    else
        _wm->_desiredRotationalVelocity = 0.;

    
    if ( TemplateEESharedData::gEnergyRequestOutput )
    {
        _wm->setEnergyRequestValue(outputs[2]);
    }
    
    // normalize to motor interval values
    _wm->_desiredTranslationalValue = _wm->_desiredTranslationalValue * gMaxTranslationalSpeed;
    _wm->_desiredRotationalVelocity = _wm->_desiredRotationalVelocity * gMaxRotationalSpeed;
    
    // * register all objects seen through distance sensors and check if caught an object (or not, then, who did?)
    // This code block is used to check if the current robot got one object it was perceiving, or if the object was "stolen" by someone else.

    bool localDebug = false;
    
    int firstCheckedObjectId = -1;
    int lastCheckedObjectId = -1; // used to consider unique object ids.
    
    int objectOnFloorIndex = _wm->getGroundSensorValue();
    if ( PhysicalObject::isInstanceOf(objectOnFloorIndex) )
    {
        objectOnFloorIndex = objectOnFloorIndex - gPhysicalObjectIndexStartOffset; // note that object may have already disappeared, though its trace remains in the floor sensor.
    }
    else
    {
        objectOnFloorIndex = -1;
    }

    PhysicalObject* lastWalkedObject = NULL;
    if ( objectOnFloorIndex != -1 )
        lastWalkedObject = gPhysicalObjects[objectOnFloorIndex];
    
    for(int i  = 0; i < _wm->_cameraSensorsNb; i++)
    {
        if ( lastSeenObjectIdPerSensorList[i] != -1 )
        {
            int targetId = lastSeenObjectIdPerSensorList[i];
            
            if ( firstCheckedObjectId == -1 )
                firstCheckedObjectId = targetId;
            
            PhysicalObject* object = gPhysicalObjects[targetId];
            
            if ( object->getTimestepSinceRelocation() == 0 )
            {
                if ( objectOnFloorIndex != -1 && object->getId() == lastWalkedObject->getId() )
                {
                    if ( targetId != lastCheckedObjectId && !( i == _wm->_cameraSensorsNb - 1 && targetId != firstCheckedObjectId ) )
                    {
                        if ( localDebug )
                            std::cout << "[DEBUG] robot #" << _wm->getId() << " says: I gathered object no." << object->getId() << "!\n";
                    }
                    else
                    {
                        if ( localDebug )
                            std::cout << "[DEBUG] robot #" << _wm->getId() << " says: I gathered object no." << object->getId() << "! (already said that)\n";
                    }
                }
                else
                {
                    if ( targetId != lastCheckedObjectId && !( i == _wm->_cameraSensorsNb - 1 && targetId != firstCheckedObjectId ) )
                    {
                        if ( localDebug )
                            std::cout << "[DEBUG] robot #" << _wm->getId() << " says: object no." << object->getId() << " disappeared! (not me!)\n";
                        this->regret += WMEESharedData::regretValue; // so frustrating!
                    }
                    else
                    {
                        if ( localDebug )
                            std::cout << "[DEBUG] robot #" << _wm->getId() << " says: object no." << object->getId() << " disappeared! (not me!) (already said that)\n";
                    }
                }
            }

            if ( lastCheckedObjectId != targetId )  // note that distance sensors cannot list obj1,obj2,obj1 due to similar object size. ie.: continuity hypothesis wrt object sensing (partial occlusion by another robot is not a problem, as this part of the code is executed only in an object is detected).
                lastCheckedObjectId = targetId;
            
        }
    }
    
    // store current sensor values for next step.
    for(int i  = 0; i < _wm->_cameraSensorsNb; i++)
    {
        int objectId = _wm->getObjectIdFromCameraSensor(i);
        
        if ( PhysicalObject::isInstanceOf(objectId) )
        {
            lastSeenObjectIdPerSensorList[i] = objectId - gPhysicalObjectIndexStartOffset;
        }
        else
        {
            lastSeenObjectIdPerSensorList[i] = -1;
        }
    }
    
}

void WMEEController::initController()
{
    TemplateEEController::initController();
}


double WMEEController::testAE(std::shared_ptr<AE> ae)
{
    torch::NoGradGuard no_grad;
    ae->get()->eval();
    size_t const dataset_size = WMEESharedData::dataBaseMaxCapacity;
    auto output = ae->get()->forward(_data);
    auto test_loss = torch::mse_loss(output.reconstruction, _data);
//    auto test_loss = torch::binary_cross_entropy(output.reconstruction, _data, {}, Reduction::Sum);
    auto sum_loss = torch::sum(test_loss);
//    //loss.backward();
    return sum_loss.item<double>();
    //return sum_loss.accessor<double, 1>()[0];
//    return 0.;
}

//template <typename DataLoader>
//void test(
//    Net& model,
//    torch::Device device,
//    DataLoader& data_loader,
//    size_t dataset_size) {
//  torch::NoGradGuard no_grad;
//  model.eval();
//  double test_loss = 0;
//  int32_t correct = 0;
//  for (const auto& batch : data_loader) {
//    auto data = batch.data.to(device), targets = batch.target.to(device);
//    auto output = model.forward(data);
//    test_loss += torch::nll_loss(
//                     output,
//                     targets,
//                     /*weight=*/{},
//                     Reduction::Sum)
//                     .template item<float>();
//    auto pred = output.argmax(1);
//    correct += pred.eq(targets).sum().template item<int64_t>();
//  }
//
//  test_loss /= dataset_size;
//  std::printf(
//      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
//      test_loss,
//      static_cast<double>(correct) / dataset_size);
//}


void WMEEController::trainAE()
{
    _visual_nn->get()->train();

    size_t const dataset_size = WMEESharedData::dataBaseMaxCapacity;
    auto output = _visual_nn->get()->forward(_data);
    //auto test_loss = torch::binary_cross_entropy(output.reconstruction, _data, {}, Reduction::Sum);
    auto test_loss = torch::mse_loss(output.reconstruction, _data);

//    double const learning_rate = 1e-9;
//    //auto optimizer = torch::optim::SGD(_visual_nn->get()->parameters(), learning_rate);
//    auto optimizer = torch::optim::Adam(_visual_nn->get()->parameters(), torch::optim::AdamOptions(learning_rate));
//    optimizer.zero_grad();
//    //std::cout << "DEBUG train: test_loss:" << _data << " " << test_loss.item<double>() << std::endl;
//    test_loss.backward();
//    optimizer.step();

    _visual_nn->get()->optim->zero_grad();
    //std::cout << "DEBUG train: test_loss:" << _data << " " << test_loss.item<double>() << std::endl;
    test_loss.backward();
    _visual_nn->get()->optim->step();
}

//template <typename DataLoader>
//void train(
//    size_t epoch,
//    Net& model,
//    torch::Device device,
//    DataLoader& data_loader,
//    torch::optim::Optimizer& optimizer,
//    size_t dataset_size) {
//  model.train();
//  size_t batch_idx = 0;
//  for (auto& batch : data_loader) {
//    auto data = batch.data.to(device), targets = batch.target.to(device);
//    optimizer.zero_grad();
//    auto output = model.forward(data);
//    auto loss = torch::nll_loss(output, targets);
//    AT_ASSERT(!std::isnan(loss.template item<float>()));
//    loss.backward();
//    optimizer.step();
//
//    if (batch_idx++ % kLogInterval == 0) {
//      std::printf(
//          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
//          epoch,
//          batch_idx * batch.data.size(0),
//          dataset_size,
//          loss.template item<float>());
//    }
//  }
//}


torch::Tensor WMEEController::mdn_loss_function(torch::Tensor pi, torch::Tensor sigma, torch::Tensor mu, torch::Tensor y) {
    float const epsilon = 1e-6;

//    auto y2 = y.expand_as(sigma);
    auto y2 = y.view({-1, WMEESharedData::dataMemNbSequences, 1, WMEESharedData::aeZDim});
    auto var = sigma * sigma;
    auto log_prob = -torch::pow(y2 - mu, 2.) / (2. * var) - var.log() - std::log(std::sqrt(2. * M_PI));
    auto res = torch::exp(log_prob);

//    std::cout << "DEBUGforward: sigma:";
//    for(size_t i = 0; i < 4; ++i)
//        std::cout << sigma.size(i) << ",";
//    std::cout << std::endl;
//
//    //auto y2 = y.unsqueeze(1).expand_as(sigma);
//    auto y2 = y.expand_as(sigma);
//    auto res = 1. / std::sqrt(2. * M_PI) * torch::exp(-0.5 * torch::pow(((y2 - mu)/sigma), 2.)) / sigma;
//    res = torch::prod(res, 2); // XXX Needed ?
//
//    std::cout << "DEBUGforward: res:";
//    for(int64_t i = 0; i < 3; ++i)
//        std::cout << res.size(i) << ",";
//    std::cout << std::endl;
//    std::cout << "DEBUGforward: pi:";
//    for(int64_t i = 0; i < 3; ++i)
//        std::cout << pi.size(i) << ",";
//    std::cout << std::endl;

    res = torch::sum(res * pi, 2);
    res = -torch::log(res + epsilon);
    //return torch::mean(res).item<double>();
    return torch::mean(res);
}


double WMEEController::testMM(std::shared_ptr<MDNLSTM> lstm) {
    torch::NoGradGuard no_grad;
    lstm->get()->eval();
    lstm->get()->reset_hidden();
    //size_t const dataset_size = WMEESharedData::dataBaseMaxCapacity;

    // Forward
    auto output = lstm->get()->forward(_data_mem);

    // Compute target
    // TODO target (same shape as sigma, but _data_mem(datasize=1000, nbsequences=2, latentvecsize+nboutputs=5+2), with only the last entry of each sequence, without the outputs)
    // Shape: (1, 1000, 2, 5) -> (?=1, datasize, nbsequences, latentvecsize)
    //auto target = output.sigma;
    auto target = torch::empty({1, WMEESharedData::dataBaseMaxCapacity, 1, WMEESharedData::aeZDim});
    auto target_access = target.accessor<float,4>();
    auto data_access = _data_mem.accessor<float,3>();
    for(int64_t i = 0; i < data_access.size(0); i++) {
        for(int64_t j = 0; j < WMEESharedData::aeZDim; j++) {
            target_access[0][i][0][j] = data_access[i][WMEESharedData::dataMemNbSequences-1][j];
        }
    }

    return mdn_loss_function(output.pi, output.sigma, output.mu, target).item<double>();
}

void WMEEController::trainMM() {
    _memory_nn->get()->train();
    // Reset gradients and hidden states
    _memory_nn->get()->optim->zero_grad();
    _memory_nn->get()->reset_hidden();

    // Forward
    auto output = _memory_nn->get()->forward(_data_mem);

    // Compute target
    auto target = torch::empty({1, WMEESharedData::dataBaseMaxCapacity, 1, WMEESharedData::aeZDim});
    auto target_access = target.accessor<float,4>();
    auto data_access = _data_mem.accessor<float,3>();
    for(int64_t i = 0; i < data_access.size(0); i++) {
        for(int64_t j = 0; j < WMEESharedData::aeZDim; j++) {
            target_access[0][i][0][j] = data_access[i][WMEESharedData::dataMemNbSequences-1][j];
        }
    }

    // Compute loss
    auto const loss = mdn_loss_function(output.pi, output.sigma, output.mu, target); // XXX
    //auto const loss = mdn_loss_function(output.pi, output.sigma, output.mu, output.sigma); // XXX
    loss.backward();
    _memory_nn->get()->optim->step();
}



void WMEEController::stepEvolution()
{
    TemplateEEController::stepEvolution();

    if(WMEESharedData::stopTrainingAfterIt > 0 && gWorld->getIterations() >= WMEESharedData::stopTrainingAfterIt) {
        _enable_memory_nn_training = false;
        _enable_visual_nn_training = false;
    }

    if( _enable_visual_nn_training && _data_idx > (size_t)WMEESharedData::dataBaseMaxCapacity && gWorld->getIterations() > 0 && gWorld->getIterations() % TemplateEESharedData::gEvaluationTime == 0 ) {
        // Add own WM NN to _vec_visual_nn
        //std::cout << "DEBUG stepEvolution _vec_visual_nn.size()=" << _vec_visual_nn.size() << std::endl;
        if (_vec_visual_nn.size() == 0) {
            //_vec_visual_nn.push_back(_visual_nn);
            _vec_visual_nn[std::make_pair(_wm->getId(), _birthdate)] = _visual_nn;
        }

        // Assess the performance of every stored world model NNs
        //std::vector<double> ae_losses;
        std::map< std::pair<int,int>, double > ae_losses;
        //std::vector<double> ae_acc;
        for(auto pae : _vec_visual_nn) {
            ae_losses[pae.first] = testAE(pae.second);
            //ae_losses.push_back(testAE(pae));
            ////auto res = testAE(pae.get());
            ////ae_losses.push_back(res.first);
            ////ae_acc.push_back(res.second);
        }
        //std::cout << "DEBUG AE losses:" << ae_losses << std::endl;

        // Select the best-performing WM NNs
        auto const best_it = std::min_element(ae_losses.begin(), ae_losses.end(), [](auto const& l, auto const& r) { return l.second < r.second; });
        //size_t const best_idx = best_it - ae_losses.begin();
        ////double const best_acc = *best_it;
        //std::cout << "DEBUG best_idx: " << best_idx << std::endl;
        //_visual_nn = _vec_visual_nn[best_idx];
        _visual_nn = _vec_visual_nn[best_it->first];
        visual_nn_loss = ae_losses[best_it->first];
        //std::cout << "DEBUG best_idx: " << best_it->first << ": " << best_it->second << std::endl;

        // Get rid of the other WM NNs
        _vec_visual_nn.clear();

        // Perform training of the world model NNs
        trainAE();
    }

    if( _enable_memory_nn_training && _data_mem_idx > (size_t)WMEESharedData::dataBaseMaxCapacity && gWorld->getIterations() > 0 && gWorld->getIterations() % TemplateEESharedData::gEvaluationTime == 0 ) {
        // Add own WM NN to _vec_memory_nn
        //std::cout << "DEBUG stepEvolution _vec_memory_nn.size()=" << _vec_memory_nn.size() << std::endl;
        if (_vec_memory_nn.size() == 0) {
            _vec_memory_nn[std::make_pair(_wm->getId(), _birthdate)] = _memory_nn;
        }

        // Assess the performance of every stored world model NNs
        std::map< std::pair<int,int>, double > mm_losses;
        for(auto pae : _vec_memory_nn) {
            mm_losses[pae.first] = testMM(pae.second);
        }
        //std::cout << "DEBUG MM losses:" << mm_losses << std::endl;

        // Select the best-performing WM NNs
        auto const best_it = std::min_element(mm_losses.begin(), mm_losses.end(), [](auto const& l, auto const& r) { return l.second < r.second; });
        _memory_nn = _vec_memory_nn[best_it->first];
        memory_nn_loss = mm_losses[best_it->first];

        // Get rid of the other WM NNs
        _vec_memory_nn.clear();

        // Perform training of the world model NNs
        trainMM();
    }

}

void WMEEController::initEvolution()
{
    TemplateEEController::initEvolution();
}

// TODO update -- remove unused methods ?
void WMEEController::performSelection()
{
    switch ( TemplateEESharedData::gSelectionMethod )
    {
        case 0:
        case 1:
        case 2:
        case 3:
            TemplateEEController::performSelection();
            break;
        case 4:
            selectNaiveMO();
            break;
        default:
            std::cerr << "[ERROR] unknown selection method (gSelectionMethod = " << TemplateEESharedData::gSelectionMethod << ")\n";
            exit(-1);
    }
}

// TODO remove ?
void WMEEController::selectNaiveMO()
{
    std::map<std::pair<int,int>, float >::iterator fitnessItUnderFocus = _fitnessValueList.begin();
    std::map<std::pair<int,int>, int >::iterator regretItUnderFocus = _regretValueList.begin();
 
    std::vector<std::pair<int,int>> paretoFrontGenomeList;
    
    // build the Pareto front
    
    for ( ; fitnessItUnderFocus != _fitnessValueList.end(); ++fitnessItUnderFocus, ++regretItUnderFocus )
    {
        std::map<std::pair<int,int>, float >::iterator fitnessItChallenger = _fitnessValueList.begin();
        std::map<std::pair<int,int>, int >::iterator regretItChallenger = _regretValueList.begin();
        
        bool candidate = true;
        
        for ( ; fitnessItChallenger != _fitnessValueList.end(); ++fitnessItChallenger, ++regretItChallenger )
        {
            if ( (*fitnessItUnderFocus).second < (*fitnessItChallenger).second and (*regretItUnderFocus).second > (*regretItChallenger).second ) // remember: regret is positive and larger value is worse.
            {
                candidate = false;
                break;
            }
        }
        if ( candidate == true )
            paretoFrontGenomeList.push_back( (*fitnessItUnderFocus).first );
    }
    
    // select a random genome from the Pareto front
    
    int randomIndex = randint()%paretoFrontGenomeList.size();
    std::pair<int,int> selectId = paretoFrontGenomeList.at(randomIndex);
    
    // update current genome with selected parent (mutation will be done elsewhere)
    
    _birthdate = gWorld->getIterations();
    
    _currentGenome = _genomesList[selectId];
    _currentSigma = _sigmaList[selectId];
    
    setNewGenomeStatus(true);
}

void WMEEController::performVariation()
{
    TemplateEEController::performVariation();
}

void WMEEController::broadcastGenome()
{
    TemplateEEController::broadcastGenome();
}

double WMEEController::getFitness()
{
    switch ( WMEESharedData::fitnessFunction )
    {
        case 0:
            return 0.0; // no fitness (ie. medea) [CTL]
            break;
        case 1: // foraging-only
//        case 4: // naive MO (using foraging and regret as seperate objectives)
            return std::abs(_wm->_fitnessValue); // foraging-only (or naive MO, which uses fitness as foraging)
            break;
//        case 2:
//            return std::max( 0.0, ( std::abs(_wm->_fitnessValue) - this->regret ) ); // foraging and regret (aggregated)
//            break;
//        case 3:
//            return -(double)this->regret; // regret-only [CTL]
//            break;
        default:
            std::cerr << "[ERROR] Fitness function unknown (check fitnessFunction value). Exiting.\n";
            exit (-1);
    }
}


void WMEEController::resetFitness()
{
    TemplateEEController::resetFitness();
    
    nbForagedItemType0 = 0;
    nbForagedItemType1 = 0;
    
    this->regret = 0;
}


void WMEEController::updateFitness()
{
    // nothing to do -- updating is performed in ForagingRegionAgentObserver (automatic event when energy item are captured)
}


void WMEEController::logCurrentState()
{
    TemplateEEController::logCurrentState();
}


bool WMEEController::sendGenome( TemplateEEController* __targetRobotController )
{
    WMEEPacket* p = new WMEEPacket();
    p->senderId = std::make_pair(_wm->getId(), _birthdate);
    p->fitness = getFitness();
    p->genome = _currentGenome;
    p->sigma = _currentSigma;
    p->regret = this->regret;
    p->visual_nn = this->_visual_nn;
    p->memory_nn = this->_memory_nn;
    
    bool retValue = ((WMEEController*)__targetRobotController)->receiveGenome(p);

    delete p;

    return retValue;
}


bool WMEEController::receiveGenome( Packet* p )
{
    WMEEPacket* p2 = static_cast<WMEEPacket*>(p);
    
    std::map<std::pair<int,int>, std::vector<double> >::const_iterator it = _genomesList.find(p2->senderId);
    
    _fitnessValueList[p2->senderId] = p2->fitness;
    _regretValueList[p2->senderId] = p2->regret;

    if (_vec_visual_nn.size() < (size_t)WMEESharedData::maxStoredVisualModels) {
        _vec_visual_nn[p2->senderId] = p2->visual_nn; // XXX
        ////_vec_visual_nn.push_back(p2->visual_nn);
        ////std::cout << "#" << _vec_visual_nn.size() << std::endl;
    }

    if (_vec_memory_nn.size() < (size_t)WMEESharedData::maxStoredMemoryModels) {
        _vec_memory_nn[p2->senderId] = p2->memory_nn; // XXX
    }
    
    if ( it == _genomesList.end() ) // this exact agent's genome is already stored. Exact means: same robot, same generation. Then: update fitness value (the rest in unchanged)
    {
        _genomesList[p2->senderId] = p2->genome;
        _sigmaList[p2->senderId] = p2->sigma;
        return true;
    }
    else
    {
        return false;
    }
}


void WMEEController::createNN()
{
    setIOcontrollerSize(); // compute #inputs and #outputs
    
    if ( nn != NULL ) // useless: delete will anyway check if nn is NULL or not.
        delete nn;

    // Create Visual Model
    _visual_nn = std::make_shared<AE>(_nbInputs, (size_t)WMEESharedData::aeHDim, (size_t)WMEESharedData::aeZDim, WMEESharedData::learningRate);
    //torch::Device device(torch::kCPU);
    _visual_nn->get()->to(torch::kCPU);
    _visual_nn->get()->train();

    // TODO
    // Create Memory Model
    _memory_nn = std::make_shared<MDNLSTM>(WMEESharedData::dataMemNbSequences, WMEESharedData::mdnlstmHDim, WMEESharedData::aeZDim, _nbOutputs, WMEESharedData::mdnlstmNbLayers, WMEESharedData::mdnlstmNbSamples, WMEESharedData::mdnlstmHiddenDim, WMEESharedData::mdnlstmTemperature, WMEESharedData::learningRateMDNLSTM);
    _visual_nn->get()->to(torch::kCPU);
    _visual_nn->get()->train();
    
    switch ( WMEESharedData::gControllerType )
    {
        case 0:
        {
            // MLP
            nn = new MLP(_parameters, _nbInputs, _nbOutputs, *(_nbNeuronsPerHiddenLayer));
            break;
        }
        case 1:
        {
            // PERCEPTRON
            nn = new Perceptron(_parameters, _nbInputs, _nbOutputs);
            break;
        }
        case 2:
        {
            // ELMAN
            nn = new Elman(_parameters, _nbInputs, _nbOutputs, *(_nbNeuronsPerHiddenLayer));
            break;
        }
        case 3:
        {
            // ESNEigen
			ESNEigen::seed_t seedESN = 0L;
            nn = new ESNEigen(_parameters, _nbInputs, _nbOutputs, TemplateEESharedData::gESNReservoirSize, TemplateEESharedData::gESNDensityOfConnections, TemplateEESharedData::gESNAlpha, seedESN, 0.5, 0.5, 0.5, 0.5, 0.5,
				TemplateEESharedData::gESNAllowInputToOutputDirectConnections,	// allowInputToOutputDirectConnections
				TemplateEESharedData::gESNAllowOutputSelfRecurrentConnections,	// allowOutputSelfRecurrentConnections
				TemplateEESharedData::gESNAllowInputToReservoirConnections,	// allowInputToReservoirConnections
				TemplateEESharedData::gESNFixedInputToReservoirConnections,	// fixedInputToReservoirConnections
				TemplateEESharedData::gESNAllowOutputToReservoirConnections, 	// allowOutputToReservoirConnections
				TemplateEESharedData::gESNAddConstantInputBias,	// addConstantInputBias 
				TemplateEESharedData::gESNAddSinInputBias,	// addSinInputBias
				TemplateEESharedData::gESNSinBiasPeriod,	// sinBiasPeriod 
				TemplateEESharedData::gESNUseSparseComputation	// useSparseComputation
			);
            break;
        }
        case 4:
        {
            // AE + MLP
            nn = new MLP(_parameters, (size_t)WMEESharedData::aeZDim, _nbOutputs, *(_nbNeuronsPerHiddenLayer));
            _enable_visual_nn_training = true;
            _enable_memory_nn_training = false;
            break;
        }
        case 5:
        {
            // TODO
            // MDNLSTM + AE + MLP
            nn = new MLP(_parameters, (size_t)WMEESharedData::aeZDim + (size_t)WMEESharedData::mdnlstmHDim, _nbOutputs, *(_nbNeuronsPerHiddenLayer));
            _enable_visual_nn_training = false;
            _enable_memory_nn_training = true;
            break;
        }

        default: // default: no controller
            std::cerr << "[ERROR] gController type unknown (value: " << WMEESharedData::gControllerType << ").\n";
            exit(-1);
    };

}




