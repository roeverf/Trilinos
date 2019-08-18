//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_RGDSWCOARSEOPERATOR_DEF_HPP
#define _FROSCH_RGDSWCOARSEOPERATOR_DEF_HPP

#include <FROSch_RGDSWCoarseOperator_decl.hpp>

namespace FROSch {

    template <class SC,class LO, class GO,class NO>
    int RGDSWCoarseOperator<SC,LO,GO,NO>::current_level = 0;
    template <class SC,class LO,class GO,class NO>
    RGDSWCoarseOperator<SC,LO,GO,NO>::RGDSWCoarseOperator(ConstCrsMatrixPtr k,
                                                          ParameterListPtr parameterList) :
    GDSWCoarseOperator<SC,LO,GO,NO> (k,parameterList)
    #ifdef FROSCH_RGDSWCOARSEOPERATOR_TIMER
    ,resetCBlockTimer(this->level),
    compTransTimer(this->level),
    compRotTimer(this->level)
    #endif
    {
      #ifdef FROSCH_RGDSWCOARSEOPERATOR_TIMER
      for(int i = 0;i<this->level;i++){
        resetCBlockTimer[i] = Teuchos::TimeMonitor::getNewCounter("FROSch RGDSWCoarseOperator resetCSBlock"+std::to_string(i));
        compTransTimer[i] = Teuchos::TimeMonitor::getNewCounter("FROSch RGDSWCoarseOperator comp Translation"+std::to_string(i));
        compRotTimer[i] = Teuchos::TimeMonitor::getNewCounter("FROSch RGDSWCoarseOperator comp Rotations"+std::to_string(i));
      }
      #endif
      current_level = current_level+1;
    }

    template <class SC,class LO,class GO,class NO>
    int RGDSWCoarseOperator<SC,LO,GO,NO>::resetCoarseSpaceBlock(UN blockId,
                                                                UN dimension,
                                                                UN dofsPerNode,
                                                                ConstMapPtr nodesMap,
                                                                ConstMapPtrVecPtr dofsMaps,
                                                                GOVecPtr dirichletBoundaryDofs,
                                                                ConstMultiVectorPtr nodeList)
    {
      #ifdef FROSCH_RGDSWCOARSEOPERATOR_TIMER
      Teuchos::TimeMonitor resetCBlockTimeMonitor(*resetCBlockTimer[current_level-1]);
      #endif

        FROSCH_ASSERT(dofsMaps.size()==dofsPerNode,"dofsMaps.size()!=dofsPerNode");
        FROSCH_ASSERT(blockId<this->NumberOfBlocks_,"Block does not exist yet and can therefore not be reset.");


        UN tra;
        UN rot;
        if (this->Verbose_) {
            std::cout << "\n\
+---------------------+\n\
| RGDSWCoarseOperator |\n\
|  Block " << blockId << "            |\n\
+---------------------+\n";
        }

        // Process the parameter list
        std::stringstream blockIdStringstream;
        blockIdStringstream << blockId+1;
        std::string blockIdString = blockIdStringstream.str();
        Teuchos::RCP<Teuchos::ParameterList> coarseSpaceList = sublist(sublist(this->ParameterList_,"Blocks"),blockIdString.c_str());

        CommunicationStrategy communicationStrategy;
        if (!coarseSpaceList->get("Interface Communication Strategy","CreateOneToOneMap").compare("CrsMatrix")) {
            communicationStrategy = CommCrsMatrix;
        } else if (!coarseSpaceList->get("Interface Communication Strategy","CreateOneToOneMap").compare("CrsGraph")) {
            communicationStrategy = CommCrsGraph;
        } else if (!coarseSpaceList->get("Interface Communication Strategy","CreateOneToOneMap").compare("CreateOneToOneMap")) {
            communicationStrategy = CreateOneToOneMap;
        } else {
            FROSCH_ASSERT(false,"FROSch::RGDSWCoarseOperator : ERROR: Specify a valid communication strategy for the identification of the interface components.");
        }

        Verbosity verbosity;
        if (!coarseSpaceList->get("Verbosity","All").compare("None")) {
            verbosity = None;
        } else if (!coarseSpaceList->get("Verbosity","All").compare("All")) {
            verbosity = All;
        } else {
            FROSCH_ASSERT(false,"FROSch::RGDSWCoarseOperator : ERROR: Specify a valid verbosity level.");
        }

        bool useForCoarseSpace = coarseSpaceList->get("Use For Coarse Space",false);
        std::string option = coarseSpaceList->get("Option","1");
        DistanceFunction distanceFunction = ConstantDistanceFunction;
        if (!option.compare("1")) {

        } else if (!option.compare("2.2")) {
            distanceFunction = InverseEuclideanDistanceFunction;
        } else {
            FROSCH_ASSERT(false,"Option is unknown!");
        }

        bool useRotations = coarseSpaceList->get("Rotations",true);
        if (useRotations && nodeList.is_null()) {
            //FROSCH_ASSERT(option==1,"Only option 1 can be constructed without a valid node list.");
            useRotations = false;
            if (this->Verbose_) std::cout << "FROSch::RGDSWCoarseOperator : WARNING: Rotations cannot be used" << std::endl;
        }

        this->DofsMaps_[blockId] = dofsMaps;
        this->DofsPerNode_[blockId] = dofsPerNode;

        Teuchos::Array<GO> tmpDirichletBoundaryDofs(dirichletBoundaryDofs()); // Here, we do a copy. Maybe, this is not necessary
        sortunique(tmpDirichletBoundaryDofs);

        this->DDInterface_.reset(new DDInterface<SC,LO,GO,NO>(dimension,this->DofsPerNode_[blockId],nodesMap.getConst(),verbosity,communicationStrategy));
        this->DDInterface_->resetGlobalDofs(dofsMaps);
        this->DDInterface_->removeDirichletNodes(tmpDirichletBoundaryDofs);
        if (this->ParameterList_->get("Test Unconnected Interface",true)) {
            this->DDInterface_->divideUnconnectedEntities(this->K_);
        }

        EntitySetPtrVecPtr entitySetVector;
        EntitySetPtr coarseNodes = this->DDInterface_->getCoarseNodes();
        EntitySetPtr interface = this->DDInterface_->getInterface();
        EntitySetPtr interior = this->DDInterface_->getInterior();

        // Check for interface
        if (this->DofsPerNode_[blockId]*interface->getEntity(0)->getNumNodes()==0) {
            this->computeVolumeFunctions(blockId,dimension,nodesMap,nodeList,interior);
        } else {
            this->GammaDofs_[blockId] = LOVecPtr(this->DofsPerNode_[blockId]*interface->getEntity(0)->getNumNodes());
            this->IDofs_[blockId] = LOVecPtr(this->DofsPerNode_[blockId]*interior->getEntity(0)->getNumNodes());
            for (UN k=0; k<this->DofsPerNode_[blockId]; k++) {
                for (UN i=0; i<interface->getEntity(0)->getNumNodes(); i++) {
                    this->GammaDofs_[blockId][this->DofsPerNode_[blockId]*i+k] = interface->getEntity(0)->getLocalDofID(i,k);
                }
                for (UN i=0; i<interior->getEntity(0)->getNumNodes(); i++) {
                    this->IDofs_[blockId][this->DofsPerNode_[blockId]*i+k] = interior->getEntity(0)->getLocalDofID(i,k);
                }
            }

            this->InterfaceCoarseSpaces_[blockId].reset(new CoarseSpace<SC,LO,GO,NO>());

            if (useForCoarseSpace) {
                this->DDInterface_->buildEntityHierarchy();

                this->DDInterface_->computeDistancesToCoarseNodes(dimension,nodeList,distanceFunction);

                /////////////////////////////////
                // Coarse Node Basis Functions //
                /////////////////////////////////
                entitySetVector = this->DDInterface_->getEntitySetVector();
                this->DDInterface_->buildEntityMaps(false,
                                                    false,
                                                    false,
                                                    false,
                                                    false,
                                                    true);

                coarseNodes->buildEntityMap(nodesMap); //Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)); coarseNodes->getEntityMap()->describe(*fancy,Teuchos::VERB_EXTREME);
                this->kRowMap_ =coarseNodes->getEntityMap();


                MultiVectorPtrVecPtr translations = this->computeTranslations(blockId,this->DDInterface_->getCoarseNodes(),entitySetVector,distanceFunction);
                tra = translations.size();
                for (UN i=0; i<translations.size(); i++) {
                    this->InterfaceCoarseSpaces_[blockId]->addSubspace(this->DDInterface_->getCoarseNodes()->getEntityMap(),translations[i]);
                }

                if (useRotations) {
                    MultiVectorPtrVecPtr rotations = this->computeRotations(blockId,dimension,nodeList,this->DDInterface_->getCoarseNodes(),entitySetVector,distanceFunction);
                    rot = rotations.size();
                    for (UN i=0; i<rotations.size(); i++) {                        this->InterfaceCoarseSpaces_[blockId]->addSubspace(this->DDInterface_->getCoarseNodes()->getEntityMap(),rotations[i]);
                    }
                }

                this->InterfaceCoarseSpaces_[blockId]->assembleCoarseSpace();
                if(useRotations){
                                    this->dofs = tra + rot;
                                }else{
                                    this->dofs = tra;
                                }
                if (this->MpiComm_->getRank() == 0) {
                    std::cout << std::boolalpha << "\n\
    ------------------------------------------------------------------------------\n\
     RGDSW coarse space\n\
    ------------------------------------------------------------------------------\n\
      Coarse nodes: translations                 --- " << true << "\n\
      Coarse nodes: rotations                    --- " << useRotations << "\n\
    ------------------------------------------------------------------------------\n" << std::noboolalpha;
                }
                if (this->ParameterList_->get("Use RepMap",false)) {
                    if (this->K_->getMap()->lib() == Xpetra::UseTpetra) {
                        this->buildGlobalGraph(this->DDInterface_);
                    }
                }
            }
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    typename RGDSWCoarseOperator<SC,LO,GO,NO>::MultiVectorPtrVecPtr RGDSWCoarseOperator<SC,LO,GO,NO>::computeTranslations(UN blockId,
                                                                                                                          EntitySetPtr coarseNodes,
                                                                                                                          EntitySetPtrVecPtr entitySetVector,
                                                                                                                          DistanceFunction distanceFunction)
    {
      #ifdef FROSCH_RGDSWCOARSEOPERATOR_TIMER
      Teuchos::TimeMonitor compTransTimeMonitor(*compTransTimer[current_level-1]);
      #endif
        MultiVectorPtrVecPtr translations(this->DofsPerNode_[blockId]);
        MapPtr serialGammaMap = Xpetra::MapFactory<LO,GO,NO>::Build(this->K_->getRangeMap()->lib(),this->GammaDofs_[blockId].size(),0,this->SerialComm_);
        for (UN i=0; i<this->DofsPerNode_[blockId]; i++) {
            if (coarseNodes->getNumEntities()>0) {
                translations[i] = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(serialGammaMap,coarseNodes->getNumEntities());
            } else {
                translations[i] = Teuchos::null;
            }
        }

        // Loop over Dofs
        for (UN k=0; k<this->DofsPerNode_[blockId]; k++) {
            // Loop over entitySetVector
            for (UN i=0; i<entitySetVector.size(); i++) {
                // Loop over entities
                for (UN j=0; j<entitySetVector[i]->getNumEntities(); j++) {
                    InterfaceEntityPtr tmpEntity = entitySetVector[i]->getEntity(j);
                    LO coarseNodeID = tmpEntity->getCoarseNodeID();
                    UN numCoarseNodes = tmpEntity->getCoarseNodes()->getNumEntities();
                    if (coarseNodeID==-1) {
                        //if (numCoarseNodes==0) std::cout << coarseNodeID << " " << numCoarseNodes << " " << tmpEntity->getAncestors()->getNumEntities() << std::endl;
                        FROSCH_ASSERT(numCoarseNodes!=0,"coarseNodeID==-1 but numCoarseNodes==0!");
                        for (UN m=0; m<numCoarseNodes; m++) {
                            InterfaceEntityPtr tmpCoarseNode = tmpEntity->getCoarseNodes()->getEntity(m);
                            LO index = tmpCoarseNode->getCoarseNodeID();
                            // Offspring: loop over nodes
                            for (UN l=0; l<tmpEntity->getNumNodes(); l++) {
                                SC value = tmpEntity->getDistanceToCoarseNode(l,m)/tmpEntity->getDistanceToCoarseNode(l,numCoarseNodes);
                                translations[k]->replaceLocalValue(tmpEntity->getGammaDofID(l,k),index,value);
                            }
                        }
                    } else {
                        // Coarse node: loop over nodes
                        for (UN l=0; l<entitySetVector[i]->getEntity(j)->getNumNodes(); l++) {
                            translations[k]->replaceLocalValue(tmpEntity->getGammaDofID(l,k),coarseNodeID,Teuchos::ScalarTraits<SC>::one());
                        }
                    }
                }
            }
        }
        return translations;
    }

    template <class SC,class LO,class GO,class NO>
    typename RGDSWCoarseOperator<SC,LO,GO,NO>::MultiVectorPtrVecPtr RGDSWCoarseOperator<SC,LO,GO,NO>::computeRotations(UN blockId,
                                                                                                                       UN dimension,
                                                                                                                       ConstMultiVectorPtr nodeList,
                                                                                                                       EntitySetPtr coarseNodes,
                                                                                                                       EntitySetPtrVecPtr entitySetVector,
                                                                                                                       DistanceFunction distanceFunction)
    {
      #ifdef FROSCH_RGDSWCOARSEOPERATOR_TIMER
      Teuchos::TimeMonitor compRotTimeMonitor(*compRotTimer[current_level-1]);
      #endif
        FROSCH_ASSERT(nodeList->getNumVectors()==dimension,"dimension of the nodeList is wrong.");
        FROSCH_ASSERT(dimension==this->DofsPerNode_[blockId],"dimension!=this->DofsPerNode_[blockId]");

        UN rotationsPerEntity = 0;
        switch (dimension) {
            case 1:
                return Teuchos::null;
                break;
            case 2:
                rotationsPerEntity = 1;
                break;
            case 3:
                rotationsPerEntity = 3;
                break;
            default:
                FROSCH_ASSERT(false,"The dimension is neither 2 nor 3!");
                break;
        }

        MultiVectorPtrVecPtr rotations(rotationsPerEntity);
        MapPtr serialGammaMap = Xpetra::MapFactory<LO,GO,NO>::Build(this->K_->getRangeMap()->lib(),this->GammaDofs_[blockId].size(),0,this->SerialComm_);
        for (UN i=0; i<rotationsPerEntity; i++) {
            if (coarseNodes->getNumEntities()>0) {
                rotations[i] = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(serialGammaMap,coarseNodes->getNumEntities());
            } else {
                rotations[i] = Teuchos::null;
            }
        }

        SC x,y,z,rx,ry,rz;
        // Loop over entitySetVector
        for (UN i=0; i<entitySetVector.size(); i++) {
            // Loop over entities
            for (UN j=0; j<entitySetVector[i]->getNumEntities(); j++) {
                InterfaceEntityPtr tmpEntity = entitySetVector[i]->getEntity(j);
                LO coarseNodeID = tmpEntity->getCoarseNodeID();
                UN numCoarseNodes = tmpEntity->getCoarseNodes()->getNumEntities();
                if (coarseNodeID==-1) {
                    FROSCH_ASSERT(numCoarseNodes!=0,"coarseNodeID==-1 but numCoarseNodes==0!");
                    for (UN m=0; m<numCoarseNodes; m++) {
                        InterfaceEntityPtr tmpCoarseNode = tmpEntity->getCoarseNodes()->getEntity(m);
                        LO index = tmpCoarseNode->getCoarseNodeID();
                        // Offspring: loop over nodes
                        for (UN l=0; l<tmpEntity->getNumNodes(); l++) {
                            SC value = tmpEntity->getDistanceToCoarseNode(l,m)/tmpEntity->getDistanceToCoarseNode(l,numCoarseNodes);

                            // Rotations
                            x = nodeList->getData(0)[tmpEntity->getLocalNodeID(l)];
                            y = nodeList->getData(1)[tmpEntity->getLocalNodeID(l)];

                            // Rotation 1
                            rx = y;
                            ry = -x;
                            rz = 0;
                            rotations[0]->replaceLocalValue(tmpEntity->getGammaDofID(l,0),index,value*rx);
                            rotations[0]->replaceLocalValue(tmpEntity->getGammaDofID(l,1),index,value*ry);
                            if (dimension == 3) {
                                z = nodeList->getData(2)[tmpEntity->getLocalNodeID(l)];

                                rotations[0]->replaceLocalValue(tmpEntity->getGammaDofID(l,2),index,value*rz);

                                // Rotation 2
                                rx = -z;
                                ry = 0;
                                rz = x;
                                rotations[1]->replaceLocalValue(tmpEntity->getGammaDofID(l,0),index,value*rx);
                                rotations[1]->replaceLocalValue(tmpEntity->getGammaDofID(l,1),index,value*ry);
                                rotations[1]->replaceLocalValue(tmpEntity->getGammaDofID(l,2),index,value*rz);

                                // Rotation 3
                                rx = 0;
                                ry = z;
                                rz = -y;
                                rotations[2]->replaceLocalValue(tmpEntity->getGammaDofID(l,0),index,value*rx);
                                rotations[2]->replaceLocalValue(tmpEntity->getGammaDofID(l,1),index,value*ry);
                                rotations[2]->replaceLocalValue(tmpEntity->getGammaDofID(l,2),index,value*rz);
                            }
                        }
                    }
                } else {
                    // Coarse node: loop over nodes
                    for (UN l=0; l<entitySetVector[i]->getEntity(j)->getNumNodes(); l++) {
                        // Rotations
                        x = nodeList->getData(0)[tmpEntity->getLocalNodeID(l)];
                        y = nodeList->getData(1)[tmpEntity->getLocalNodeID(l)];

                        // Rotation 1
                        rx = y;
                        ry = -x;
                        rz = 0;
                        rotations[0]->replaceLocalValue(tmpEntity->getGammaDofID(l,0),coarseNodeID,rx);
                        rotations[0]->replaceLocalValue(tmpEntity->getGammaDofID(l,1),coarseNodeID,ry);
                        if (dimension == 3) {
                            z = nodeList->getData(2)[tmpEntity->getLocalNodeID(l)];

                            rotations[0]->replaceLocalValue(tmpEntity->getGammaDofID(l,2),coarseNodeID,rz);

                            // Rotation 2
                            rx = -z;
                            ry = 0;
                            rz = x;
                            rotations[1]->replaceLocalValue(tmpEntity->getGammaDofID(l,0),coarseNodeID,rx);
                            rotations[1]->replaceLocalValue(tmpEntity->getGammaDofID(l,1),coarseNodeID,ry);
                            rotations[1]->replaceLocalValue(tmpEntity->getGammaDofID(l,2),coarseNodeID,rz);

                            // Rotation 3
                            rx = 0;
                            ry = z;
                            rz = -y;
                            rotations[2]->replaceLocalValue(tmpEntity->getGammaDofID(l,0),coarseNodeID,rx);
                            rotations[2]->replaceLocalValue(tmpEntity->getGammaDofID(l,1),coarseNodeID,ry);
                            rotations[2]->replaceLocalValue(tmpEntity->getGammaDofID(l,2),coarseNodeID,rz);
                        }
                    }
                }
            }
        }
        return rotations;
    }
}

#endif
