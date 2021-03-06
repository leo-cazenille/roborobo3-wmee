/**
 * @author Nicolas Bredeche <nicolas.bredeche@upmc.fr>
 *
 */

#include "WMEE/include/WMEEWorldObserver.h"
#include "WMEE/include/WMEEController.h"
#include "WMEE/include/WMEESharedData.h"
#include "WMEE/include/WMEEEnergyItem.h"
#include "WorldModels/RobotWorldModel.h"
#include "World/World.h"
#include "RoboroboMain/roborobo.h"
#include <math.h>
#include "Utilities/Misc.h"

WMEEWorldObserver::WMEEWorldObserver( World* world ) : TemplateEEWorldObserver( world )
{
    // superclass constructor called before

    gProperties.checkAndGetPropertyValue("gNbObjectsType1",&WMEESharedData::nbObjectsType1,true);
    gProperties.checkAndGetPropertyValue("gNbObjectsType2",&WMEESharedData::nbObjectsType2,true);
    gProperties.checkAndGetPropertyValue("gForagingTask",&WMEESharedData::foragingTask,true);

    gProperties.checkAndGetPropertyValue("fitnessFunction",&WMEESharedData::fitnessFunction,true);
    gProperties.checkAndGetPropertyValue("regretValue",&WMEESharedData::regretValue,true);

    gProperties.checkAndGetPropertyValue("gControllerType",&WMEESharedData::gControllerType,true);
    gProperties.checkAndGetPropertyValue("rebirthDelay",&WMEESharedData::rebirthDelay,false);
    gProperties.checkAndGetPropertyValue("dataBaseMaxCapacity",&WMEESharedData::dataBaseMaxCapacity,true);
    gProperties.checkAndGetPropertyValue("dataMemNbSequences",&WMEESharedData::dataMemNbSequences,true);
    gProperties.checkAndGetPropertyValue("maxStoredVisualModels",&WMEESharedData::maxStoredVisualModels,true);
    gProperties.checkAndGetPropertyValue("learningRate",&WMEESharedData::learningRate,true);
    gProperties.checkAndGetPropertyValue("learningRateMDNLSTM",&WMEESharedData::learningRateMDNLSTM,true);
    gProperties.checkAndGetPropertyValue("phase2AfterIt",&WMEESharedData::phase2AfterIt,true);
    gProperties.checkAndGetPropertyValue("phase2ControllerType",&WMEESharedData::phase2ControllerType,true);
    gProperties.checkAndGetPropertyValue("phase3AfterIt",&WMEESharedData::phase3AfterIt,true);
    gProperties.checkAndGetPropertyValue("phase3ControllerType",&WMEESharedData::phase3ControllerType,true);
    gProperties.checkAndGetPropertyValue("aeHDim",&WMEESharedData::aeHDim,true);
    gProperties.checkAndGetPropertyValue("aeZDim",&WMEESharedData::aeZDim,true);
    gProperties.checkAndGetPropertyValue("mdnlstmHDim",&WMEESharedData::mdnlstmHDim,true);
    gProperties.checkAndGetPropertyValue("mdnlstmNbSamples",&WMEESharedData::mdnlstmNbSamples,true);
    gProperties.checkAndGetPropertyValue("mdnlstmHiddenDim",&WMEESharedData::mdnlstmHiddenDim,true);
    gProperties.checkAndGetPropertyValue("mdnlstmNbLayers",&WMEESharedData::mdnlstmNbLayers,true);
    gProperties.checkAndGetPropertyValue("mdnlstmTemperature",&WMEESharedData::mdnlstmTemperature,true);
    gProperties.checkAndGetPropertyValue("stopTrainingAfterIt",&WMEESharedData::stopTrainingAfterIt,false);

    gLitelogManager->write("# lite logger\n");
    gLitelogManager->write("# [0]:generation,[1]:iteration,[2]:populationSize,[3]:minFitness,[4]:maxFitness,[5]:avgFitnessNormalized,[6]:sumOfFitnesses,[7]:foragingBalance,[8]:avg_countForagedItemType0,[9]:stddev_countForagedItemType0, [10]:avg_countForagedItemType1,[11]:stddev_countForagedItemType1,[12]:globalWelfare,[13]minGenomeReservoirSize,[14]maxGenomeReservoirSize,[15]avgGenomeReservoirSize,[16]avgForagingBalancePerRobot,[17]stdForagingBalancePerRobot,[18]activeCountWithForaging,[19]minVisualModelLosses,[20]maxVisualModelLosses,[21]avgVisualModelLosses,[22]minMemoryModelLosses,[23]maxMemoryModelLosses,[24]avgMemoryModelLosses.\n");
    gLitelogManager->flush();
}

WMEEWorldObserver::~WMEEWorldObserver()
{
    // superclass destructor called before
}

void WMEEWorldObserver::initPre()
{
    TemplateEEWorldObserver::initPre();

    int nbObjectsTotal = WMEESharedData::nbObjectsType1 + WMEESharedData::nbObjectsType2;
    int nbObjectsInLeftRegion = WMEESharedData::nbObjectsType1;
    int threshold = WMEESharedData::nbObjectsType1;

    for ( int i = 0 ; i < nbObjectsTotal ; i++ )
    {
        // * create a new (custom) object

        int id = PhysicalObjectFactory::getNextId();
        WMEEEnergyItem *object = new WMEEEnergyItem(id);
        gPhysicalObjects.push_back( object );

        switch ( WMEESharedData::foragingTask )
        {
            case 0:
            {
                if ( i < nbObjectsInLeftRegion ) // proportion in left/right parts of environment
                    object->setRegion(0,0.5); // left part of the arena
                else
                    object->setRegion(0.5,0.5); // right part of the arena
            }
            break;

            case 1:
            case 2:
            case 3:
            {
                object->setRegion(0,1); // whole arena
                if ( i < threshold )
                {
                    object->setDisplayColor(255,128,64); // orange
                    object->setType(0);
                }
                else
                {
                    object->setDisplayColor(64,192,255); // blue
                    object->setType(1);
                }
            }
            break;

            default:
            std::cerr << "[ERROR] gForagingTask value is unkown. Exiting.\n";
            break;
        }

        object->relocate();
    }
}

void WMEEWorldObserver::initPost()
{
    TemplateEEWorldObserver::initPost();

    gNbOfPhysicalObjects = (int)gPhysicalObjects.size(); // must be done in the initPost() method for objects created in initPre().
}

void WMEEWorldObserver::stepPre()
{
    TemplateEEWorldObserver::stepPre();
    /*
    // EXAMPLE
    if( gWorld->getIterations() > 0 && gWorld->getIterations() % TemplateEESharedData::gEvaluationTime == 0 )
    {
        std::cout << "[INFO] new generation.\n";
    }
    */
}

void WMEEWorldObserver::stepPost( )
{
    TemplateEEWorldObserver::stepPost();
}

void WMEEWorldObserver::monitorPopulation( bool localVerbose )
{
    //TemplateEEWorldObserver::monitorPopulation(localVerbose);

    // * monitoring: count number of active agents.

    int activeCount = 0;
    int activeCountWithForaging = 0; // count active robots that forage at least one item.

    double sumOfFitnesses = 0;
    double minFitness = DBL_MAX;
    double maxFitness = -DBL_MAX;

    int countForagedItemType0 = 0;
    int countForagedItemType1 = 0;

    int minGenomeReservoirSize = -1;
    int maxGenomeReservoirSize = -1;
    double avgGenomeReservoirSize = -1;

    double avgForagingBalancePerRobot = 0;
    double stddev_activeCountWithForaging = 0;

    double sumVisualModelLosses = 0.;
    double minVisualModelLosses = DBL_MAX;
    double maxVisualModelLosses = -DBL_MAX;
    double sumMemoryModelLosses = 0.;
    double minMemoryModelLosses = DBL_MAX;
    double maxMemoryModelLosses = -DBL_MAX;

    for ( int i = 0 ; i != gNbOfRobots ; i++ )
    {
        WMEEController *ctl = dynamic_cast<WMEEController*>(gWorld->getRobot(i)->getController());

        if ( ctl->getWorldModel()->isAlive() == true )
        {
            activeCount++;

            // fitnesses

            sumOfFitnesses += ctl->getFitness() ;
            if ( ctl->getFitness() < minFitness )
                minFitness = ctl->getFitness();
            if ( ctl->getFitness() > maxFitness )
                maxFitness = ctl->getFitness();

            // foraging scores

            countForagedItemType0 += ctl->nbForagedItemType0;
            countForagedItemType1 += ctl->nbForagedItemType1;

            // balancing between resources, agent-level

            if ( ctl->nbForagedItemType0 + ctl->nbForagedItemType1 > 0 )
            {
                avgForagingBalancePerRobot += getBalance(ctl->nbForagedItemType0,ctl->nbForagedItemType1);
                activeCountWithForaging++;
            }

            // genome reservoir sizes

            int genomeReservoirSize = ctl->getGenomeReservoirSize();

            avgGenomeReservoirSize += (double)genomeReservoirSize;

            if ( minGenomeReservoirSize == -1 )
            {
                minGenomeReservoirSize = genomeReservoirSize;
                maxGenomeReservoirSize = genomeReservoirSize;
            }
            else
            {
                if ( minGenomeReservoirSize > genomeReservoirSize )
                    minGenomeReservoirSize = genomeReservoirSize;
                else
                    if ( maxGenomeReservoirSize < genomeReservoirSize )
                        maxGenomeReservoirSize = genomeReservoirSize;
            }

            // Losses
            sumVisualModelLosses += ctl->visual_nn_loss;
            if(minVisualModelLosses > ctl->visual_nn_loss)
                minVisualModelLosses = ctl->visual_nn_loss;
            if(maxVisualModelLosses < ctl->visual_nn_loss)
                maxVisualModelLosses = ctl->visual_nn_loss;
            sumMemoryModelLosses += ctl->memory_nn_loss;
            if(minMemoryModelLosses > ctl->memory_nn_loss)
                minMemoryModelLosses = ctl->memory_nn_loss;
            if(maxMemoryModelLosses < ctl->memory_nn_loss)
                maxMemoryModelLosses = ctl->memory_nn_loss;
        }
    }

    avgGenomeReservoirSize = avgGenomeReservoirSize / activeCount;

    avgForagingBalancePerRobot = avgForagingBalancePerRobot / activeCountWithForaging; // robot-level, consider active robots with foraging activity.

    double foragingBalance = getBalance( countForagedItemType0 , countForagedItemType1 ); // pop-level

    double avgFitnessNormalized;

    if ( activeCount == 0 ) // arbitrary convention: in case of extinction, min/max/avg fitness values are -1
    {
        minFitness = -1;
        maxFitness = -1;
        avgFitnessNormalized = -1;
    }
    else
        avgFitnessNormalized = sumOfFitnesses/activeCount;

    double const avgVisualModelLosses = sumVisualModelLosses / activeCount;
    double const avgMemoryModelLosses = sumMemoryModelLosses / activeCount;

    // display lightweight logs for easy-parsing
    std::string sLitelog =
    "log:"
    + std::to_string(gWorld->getIterations()/TemplateEESharedData::gEvaluationTime)
    + ","
    + std::to_string(gWorld->getIterations())
    + ","
    + std::to_string(activeCount)
    + ","
    + std::to_string(minFitness)
    + ","
    + std::to_string(maxFitness)
    + ","
    + std::to_string(avgFitnessNormalized)
    + ","
    + std::to_string(sumOfFitnesses)
    + ","
    + std::to_string(foragingBalance);


    double avg_countForagedItemType0 = (double)countForagedItemType0 / activeCount;
    double avg_countForagedItemType1 = (double)countForagedItemType1 / activeCount;

    double stddev_countForagedItemType0 = 0;
    double stddev_countForagedItemType1 = 0;

    for ( int i = 0 ; i != gNbOfRobots ; i++ )
    {
        WMEEController *ctl = dynamic_cast<WMEEController*>(gWorld->getRobot(i)->getController());

        if ( ctl->getWorldModel()->isAlive() == true )
        {
            stddev_countForagedItemType0 += pow( (double)ctl->nbForagedItemType0 - avg_countForagedItemType0, 2);
            stddev_countForagedItemType1 += pow( (double)ctl->nbForagedItemType1 - avg_countForagedItemType1, 2);

            if ( ctl->nbForagedItemType0 + ctl->nbForagedItemType1 > 0 )
                stddev_activeCountWithForaging += pow( getBalance(ctl->nbForagedItemType0,ctl->nbForagedItemType1) - avgForagingBalancePerRobot, 2);
        }
    }

    stddev_countForagedItemType0 /= activeCount;
    stddev_countForagedItemType1 /= activeCount;

    stddev_activeCountWithForaging /= avgForagingBalancePerRobot;

    sLitelog += ","
    + std::to_string(avg_countForagedItemType0)
    + ","
    + std::to_string(stddev_countForagedItemType0)
    + ","
    + std::to_string(avg_countForagedItemType1)
    + ","
    + std::to_string(stddev_countForagedItemType1)
    + ","
    + std::to_string(countForagedItemType0+countForagedItemType1); // ie. global welfare, or: population-level foraging score

    sLitelog += ","
    + std::to_string(minGenomeReservoirSize)
    + ","
    + std::to_string(maxGenomeReservoirSize)
    + ","
    + std::to_string(avgGenomeReservoirSize);

    sLitelog += ","
    + std::to_string(avgForagingBalancePerRobot)
    + ","
    + std::to_string(stddev_activeCountWithForaging)
    + ","
    + std::to_string(activeCountWithForaging);

    // Losses
    sLitelog += ","
    + std::to_string(minVisualModelLosses)
    + ","
    + std::to_string(maxVisualModelLosses)
    + ","
    + std::to_string(avgVisualModelLosses)
    + ","
    + std::to_string(minMemoryModelLosses)
    + ","
    + std::to_string(maxMemoryModelLosses)
    + ","
    + std::to_string(avgMemoryModelLosses);

    gLitelogManager->write(sLitelog);
    gLitelogFile << std::endl; // flush file output (+ "\n")
    gLitelogManager->flush();  // flush internal buffer to file

    // Logging, population-level: alive
    std::string sLog = std::string("") + std::to_string(gWorld->getIterations()) + ",pop,alive," + std::to_string(activeCount) + "\n";
    gLogManager->write(sLog);
    gLogManager->flush();

    if ( gVerbose && localVerbose )
    {
        //std::cout << "[ gen:" << (gWorld->getIterations()/TemplateEESharedData::gEvaluationTime) << "\tit:" << gWorld->getIterations() << "\tpop:" << activeCount << "\tminFitness:" << minFitness << "\tmaxFitness:" << maxFitness << "\tavgFitnessNormalized:" << avgFitnessNormalized << "\tglobalWelfare:" << (countForagedItemType0+countForagedItemType1) << " ]\n";
        std::cout << "[ gen:" << (gWorld->getIterations()/TemplateEESharedData::gEvaluationTime) << "\tit:" << gWorld->getIterations() << "\tpop:" << activeCount << "\tminFitness:" << minFitness << "\tmaxFitness:" << maxFitness << "\tavgFitnessNormalized:" << avgFitnessNormalized << "\tglobalWelfare:" << (countForagedItemType0+countForagedItemType1) << "\tminVisualModelLosses:" << minVisualModelLosses << "\tmaxVisualModelLosses:" << maxVisualModelLosses << "\tavgVisualModelLosses:" << avgVisualModelLosses << "\tminMemoryModelLosses:" << minMemoryModelLosses << "\tmaxMemoryModelLosses:" << maxMemoryModelLosses << "\tavgMemoryModelLosses:" << avgMemoryModelLosses << " ]\n";
    }

}


