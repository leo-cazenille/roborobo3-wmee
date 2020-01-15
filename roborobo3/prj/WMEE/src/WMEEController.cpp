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
}


void WMEEController::step() // handles control decision and evolution (but: actual movement is done in roborobo's main loop)
{
    _iteration++;

    // Update controller type, if necessary
    if (WMEESharedData::phase2AfterIt == _iteration) {
        //std::cout << "PHASE CHANGE !!" << std::endl;
        WMEESharedData::gControllerType = WMEESharedData::phase2ControllerType;
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

    auto idx = _dataIdx % (size_t)WMEESharedData::dataBaseMaxCapacity;
    auto data_access = _data.accessor<float,2>();
    for(int j = 0; j < data_access.size(1); ++j) {
        if(isnormal(inputs[j]))
            data_access[idx][j] = (1.0 + inputs[j]) / 2.0; // Normalization from [-1, 1] to [0, 1]
        else
            data_access[idx][j] = 0.0;
    }
    ++_dataIdx;

    switch ( WMEESharedData::gControllerType )
    {
        case 4: // AE+MLP
        {
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
            //std::cout << "Z: " << z << std::endl;
            //std::vector<double> z(z_tensor.data<double>());

            //// TODO : compute Vision and memory models, use their outputs as inputs of ``nn``
            nn->setInputs(z);
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


void WMEEController::stepEvolution()
{
    TemplateEEController::stepEvolution();

    if( _dataIdx > (size_t)WMEESharedData::dataBaseMaxCapacity && gWorld->getIterations() > 0 && gWorld->getIterations() % TemplateEESharedData::gEvaluationTime == 0 ) {
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
        //std::cout << "DEBUG losses:" << ae_losses << std::endl;

        // Select the best-performing WM NNs
        auto const best_it = std::min_element(ae_losses.begin(), ae_losses.end(), [](auto const& l, auto const& r) { return l.second < r.second; });
        //size_t const best_idx = best_it - ae_losses.begin();
        ////double const best_acc = *best_it;
        //std::cout << "DEBUG best_idx: " << best_idx << std::endl;
        //_visual_nn = _vec_visual_nn[best_idx];
        _visual_nn = _vec_visual_nn[best_it->first];
        //std::cout << "DEBUG best_idx: " << best_it->first << ": " << best_it->second << std::endl;

        // Get rid of the other WM NNs
        _vec_visual_nn.clear();

        // Perform training of the world model NNs
        trainAE();
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

// TODO handle vision and memory models
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


// TODO handle vision and memory models
bool WMEEController::sendGenome( TemplateEEController* __targetRobotController )
{
    WMEEPacket* p = new WMEEPacket();
    p->senderId = std::make_pair(_wm->getId(), _birthdate);
    p->fitness = getFitness();
    p->genome = _currentGenome;
    p->sigma = _currentSigma;
    p->regret = this->regret;
    p->visual_nn = this->_visual_nn;
    
    bool retValue = ((WMEEController*)__targetRobotController)->receiveGenome(p);

    delete p;

    return retValue;
}


// TODO handle vision and memory models
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


// TODO ADD autoencoder; Vision and memory models
void WMEEController::createNN()
{
    setIOcontrollerSize(); // compute #inputs and #outputs
    
    if ( nn != NULL ) // useless: delete will anyway check if nn is NULL or not.
        delete nn;

    _visual_nn = std::make_shared<AE>(_nbInputs, (size_t)WMEESharedData::aeHDim, (size_t)WMEESharedData::aeZDim, WMEESharedData::learningRate);
    //torch::Device device(torch::kCPU);
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
            break;
        }
        default: // default: no controller
            std::cerr << "[ERROR] gController type unknown (value: " << WMEESharedData::gControllerType << ").\n";
            exit(-1);
    };

}




