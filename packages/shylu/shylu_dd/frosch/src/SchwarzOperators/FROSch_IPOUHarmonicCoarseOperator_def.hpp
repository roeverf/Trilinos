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

#ifndef _FROSCH_IPOUHARMONICCOARSEOPERATOR_DEF_HPP
#define _FROSCH_IPOUHARMONICCOARSEOPERATOR_DEF_HPP

#include <FROSch_IPOUHarmonicCoarseOperator_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::IPOUHarmonicCoarseOperator(ConstXMatrixPtr k,
                                                                         ParameterListPtr parameterList) :
    HarmonicCoarseOperator<SC,LO,GO,NO> (k,parameterList),
    PartitionOfUnity_ (),
    LocalPartitionOfUnityBasis_ ()
    {
        FROSCH_TIMER_START_LEVELID(iPOUHarmonicCoarseOperatorTime,"IPOUHarmonicCoarseOperator::IPOUHarmonicCoarseOperator");
    }

    template <class SC,class LO,class GO,class NO>
    int IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                            UN dofsPerNode,
                                                            ConstXMapPtr nodesMap,
                                                            ConstXMapPtrVecPtr dofsMaps,
                                                            ConstXMultiVectorPtr nullSpaceBasis,
                                                            ConstXMultiVectorPtr nodeList,
                                                            GOVecPtr dirichletBoundaryDofs)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"IPOUHarmonicCoarseOperator::initialize");
        int ret = buildCoarseSpace(dimension,dofsPerNode,nodesMap,dofsMaps,nullSpaceBasis,dirichletBoundaryDofs,nodeList);
        this->CoarseMap_ = this->assembleCoarseMap();
        this->buildCoarseSolveMap();
        //buildElementNodeList();
        //buildCoarseGraph();
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return ret;
    }

    template <class SC,class LO,class GO,class NO>
    int IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::initialize(UN dimension,
                                                            UNVecPtr dofsPerNodeVec,
                                                            ConstXMapPtrVecPtr repeatedNodesMapVec,
                                                            ConstXMapPtrVecPtr2D repeatedDofMapsVec,
                                                            ConstXMultiVectorPtrVecPtr nullSpaceBasisVec,
                                                            ConstXMultiVectorPtrVecPtr nodeListVec,
                                                            GOVecPtr2D dirichletBoundaryDofsVec)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"IPOUHarmonicCoarseOperator::initialize");
        buildCoarseSpace(dimension,dofsPerNodeVec,repeatedNodesMapVec,repeatedDofMapsVec,nullSpaceBasisVec,dirichletBoundaryDofsVec,nodeListVec);
        this->CoarseMap_ = this->assembleCoarseMap();
        this->buildCoarseSolveMap();
        //buildElementNodeList();
        //buildCoarseGraph();
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    void IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::describe(FancyOStream &out,
                                                           const EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has to be implemented properly...");
    }

    template <class SC,class LO,class GO,class NO>
    std::string IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::description() const
    {
        return "Interface Partition of Unity Coarse Operator";
    }

    template <class SC,class LO,class GO,class NO>
    int  IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                                   UN dofsPerNode,
                                                                   ConstXMapPtr nodesMap,
                                                                   ConstXMapPtrVecPtr dofsMaps,
                                                                   ConstXMultiVectorPtr nullSpaceBasis,
                                                                   GOVecPtr dirichletBoundaryDofs,
                                                                   ConstXMultiVectorPtr nodeList)
    {
        FROSCH_TIMER_START_LEVELID(buildCoarseSpaceTime,"IPOUHarmonicCoarseOperator::buildCoarseSpace");
        FROSCH_ASSERT(dofsMaps.size()==dofsPerNode,"dofsMaps.size()!=dofsPerNode");

        // Das könnte man noch ändern
        // LÄNGEN NOCHMAL GEGEN NumberOfBlocks_ checken!!!
        this->GammaDofs_.resize(this->GammaDofs_.size()+1);
        this->IDofs_.resize(this->IDofs_.size()+1);
        this->InterfaceCoarseSpaces_.resize(this->InterfaceCoarseSpaces_.size()+1);
        this->CoarseNullSpace_.resize(1);
        this->DofsMaps_.resize(this->DofsMaps_.size()+1);
        this->DofsPerNode_.resize(this->DofsPerNode_.size()+1);
        this->NumberOfBlocks_++;

        return resetCoarseSpaceBlock(this->NumberOfBlocks_-1,dimension,dofsPerNode,nodesMap,dofsMaps,nullSpaceBasis,dirichletBoundaryDofs,nodeList);
    }

    template <class SC,class LO,class GO,class NO>
    int IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::buildCoarseSpace(UN dimension,
                                                                  UNVecPtr dofsPerNodeVec,
                                                                  ConstXMapPtrVecPtr repeatedNodesMapVec,
                                                                  ConstXMapPtrVecPtr2D repeatedDofMapsVec,
                                                                  ConstXMultiVectorPtrVecPtr nullSpaceBasisVec,
                                                                  GOVecPtr2D dirichletBoundaryDofsVec,
                                                                  ConstXMultiVectorPtrVecPtr nodeListVec)
    {
        FROSCH_TIMER_START_LEVELID(buildCoarseSpaceTime,"IPOUHarmonicCoarseOperator::buildCoarseSpace");
        // Das könnte man noch ändern
        // TODO: DAS SOLLTE ALLES IN EINE FUNKTION IN HARMONICCOARSEOPERATOR
        for (UN i=0; i<repeatedNodesMapVec.size(); i++) {
            this->GammaDofs_.resize(this->GammaDofs_.size()+1);
            this->IDofs_.resize(this->IDofs_.size()+1);
            this->InterfaceCoarseSpaces_.resize(this->InterfaceCoarseSpaces_.size()+1);
            this->CoarseNullSpace_.resize(dofsPerNodeVec.size());
            this->DofsMaps_.resize(this->DofsMaps_.size()+1);
            this->DofsPerNode_.resize(this->DofsPerNode_.size()+1);
            this->NumberOfBlocks_++;
            resetCoarseSpaceBlock(this->NumberOfBlocks_-1,dimension,dofsPerNodeVec[i],repeatedNodesMapVec[i],repeatedDofMapsVec[i],nullSpaceBasisVec[i],dirichletBoundaryDofsVec[i],nodeListVec[i]);
        }
        return 0;
    }


    template <class SC,class LO,class GO,class NO>
    int IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::resetCoarseSpaceBlock(UN blockId,
                                                                       UN dimension,
                                                                       UN dofsPerNode,
                                                                       ConstXMapPtr nodesMap,
                                                                       ConstXMapPtrVecPtr dofsMaps,
                                                                       ConstXMultiVectorPtr nullSpaceBasis,
                                                                       GOVecPtr dirichletBoundaryDofs,
                                                                       ConstXMultiVectorPtr nodeList)
    {
        FROSCH_TIMER_START_LEVELID(resetCoarseSpaceBlockTime,"IPOUHarmonicCoarseOperator::resetCoarseSpaceBlock");
        FROSCH_ASSERT(dofsMaps.size()==dofsPerNode,"dofsMaps.size()!=dofsPerNode");
        FROSCH_ASSERT(blockId<this->NumberOfBlocks_,"Block does not exist yet and can therefore not be reset.");
        this->dofs = nullSpaceBasis->getNumVectors();
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::VerboseObjectBase::getDefaultOStream();

        if (this->Verbose_) {
            std::cout << "\n\
+----------------------------+\n\
| IPOUHarmonicCoarseOperator |\n\
|  Block " << blockId << "                   |\n\
+----------------------------+\n";
        }

        // Process the parameter list
        std::stringstream blockIdStringstream;
        blockIdStringstream << blockId+1;
        std::string blockIdString = blockIdStringstream.str();
        RCP<ParameterList> coarseSpaceList = sublist(sublist(this->ParameterList_,"Blocks"),blockIdString.c_str());

        Verbosity verbosity = All;
        if (!coarseSpaceList->get("Verbosity","All").compare("None")) {
            verbosity = None;
        } else if (!coarseSpaceList->get("Verbosity","All").compare("All")) {
            verbosity = All;
        } else {
            FROSCH_ASSERT(false,"FROSch::IPOUHarmonicCoarseOperator : ERROR: Specify a valid verbosity level.");
        }

        bool useForCoarseSpace = coarseSpaceList->get("Use For Coarse Space",true);

        if (useForCoarseSpace) {
            this->DofsMaps_[blockId] = dofsMaps;
            this->DofsPerNode_[blockId] = dofsPerNode;

            // Compute Interface Partition of Unity
            InterfacePartitionOfUnityPtr interfacePartitionOfUnity;
            if (!coarseSpaceList->sublist("InterfacePartitionOfUnity").get("Type","GDSW").compare("GDSW")) {
                coarseSpaceList->sublist("InterfacePartitionOfUnity").sublist("GDSW").set("Test Unconnected Interface",this->ParameterList_->get("Test Unconnected Interface",true));
                interfacePartitionOfUnity = InterfacePartitionOfUnityPtr(new GDSWInterfacePartitionOfUnity<SC,LO,GO,NO>(this->MpiComm_,this->SerialComm_,dimension,this->DofsPerNode_[blockId],nodesMap,this->DofsMaps_[blockId],sublist(sublist(coarseSpaceList,"InterfacePartitionOfUnity"),"GDSW"),verbosity,this->LevelID_));
            } else if (!coarseSpaceList->sublist("InterfacePartitionOfUnity").get("Type","GDSW").compare("GDSWStar")) {
                coarseSpaceList->sublist("InterfacePartitionOfUnity").sublist("GDSWStar").set("Test Unconnected Interface",this->ParameterList_->get("Test Unconnected Interface",true));
                interfacePartitionOfUnity = InterfacePartitionOfUnityPtr(new GDSWStarInterfacePartitionOfUnity<SC,LO,GO,NO>(this->MpiComm_,this->SerialComm_,dimension,this->DofsPerNode_[blockId],nodesMap,this->DofsMaps_[blockId],sublist(sublist(coarseSpaceList,"InterfacePartitionOfUnity"),"GDSWStar"),verbosity,this->LevelID_));
            } else if (!coarseSpaceList->sublist("InterfacePartitionOfUnity").get("Type","GDSW").compare("RGDSW")) {
                coarseSpaceList->sublist("InterfacePartitionOfUnity").sublist("RGDSW").set("Test Unconnected Interface",this->ParameterList_->get("Test Unconnected Interface",true));
                interfacePartitionOfUnity = InterfacePartitionOfUnityPtr(new RGDSWInterfacePartitionOfUnity<SC,LO,GO,NO>(this->MpiComm_,this->SerialComm_,dimension,this->DofsPerNode_[blockId],nodesMap,this->DofsMaps_[blockId],sublist(sublist(coarseSpaceList,"InterfacePartitionOfUnity"),"RGDSW"),verbosity,this->LevelID_));
            } else {
                FROSCH_ASSERT(false,"InterfacePartitionOfUnity Type is unknown.");
            }

            // Extract the interface and the interior from the DDInterface stored in the Interface Partition of Unity object
            InterfaceEntityPtr interface = interfacePartitionOfUnity->getDDInterface()->getInterface()->getEntity(0);
            InterfaceEntityPtr interior = interfacePartitionOfUnity->getDDInterface()->getInterior()->getEntity(0);

            // Check whether the interface is empty. If so, we use a ConstantPartitionOfUnity instead, because we assume that the subdomains are decoupled.
            if (interface->getNumNodes()==0) {
                if (this->Verbose_) std::cout << "FROSch::IPOUHarmonicCoarseOperator : WARNING: No interface found => A Constant Partition of Unity will be used instead." << std::endl;
                coarseSpaceList->sublist("InterfacePartitionOfUnity").sublist("RGDSW").set("Test Unconnected Interface",this->ParameterList_->get("Test Unconnected Interface",true));
                PartitionOfUnity_ = PartitionOfUnityPtr(new ConstantPartitionOfUnity<SC,LO,GO,NO>(this->MpiComm_,
                                                                                                  this->SerialComm_,
                                                                                                  dimension,
                                                                                                  this->DofsPerNode_[blockId],
                                                                                                  nodesMap,
                                                                                                  this->DofsMaps_[blockId],
                                                                                                  sublist(sublist(coarseSpaceList,"PartitionOfUnity"),"Constant"),
                                                                                                  verbosity,
                                                                                                  this->LevelID_,
                                                                                                  interfacePartitionOfUnity->getDDInterfaceNonConst()));

                PartitionOfUnity_->removeDirichletNodes(dirichletBoundaryDofs());

                // Construct Interface and Interior index sets
                this->GammaDofs_[blockId] = LOVecPtr(this->DofsPerNode_[blockId]*interior->getNumNodes());
                this->IDofs_[blockId] = LOVecPtr(0);
                for (UN k=0; k<this->DofsPerNode_[blockId]; k++) {
                    for (UN i=0; i<interior->getNumNodes(); i++) {
                        this->GammaDofs_[blockId][interior->getGammaDofID(i,k)] = interior->getLocalDofID(i,k);
                    }
                }

                PartitionOfUnity_->computePartitionOfUnity(nodeList);
            } else {
                interfacePartitionOfUnity->removeDirichletNodes(dirichletBoundaryDofs(),nodeList);
                interfacePartitionOfUnity->sortInterface(this->K_,nodeList);

                // Construct Interface and Interior index sets
                this->GammaDofs_[blockId] = LOVecPtr(this->DofsPerNode_[blockId]*interface->getNumNodes());
                this->IDofs_[blockId] = LOVecPtr(this->DofsPerNode_[blockId]*interior->getNumNodes());
                for (UN k=0; k<this->DofsPerNode_[blockId]; k++) {
                    for (UN i=0; i<interface->getNumNodes(); i++) {
                        this->GammaDofs_[blockId][interface->getGammaDofID(i,k)] = interface->getLocalDofID(i,k);
                    }
                    for (UN i=0; i<interior->getNumNodes(); i++) {
                        this->IDofs_[blockId][interior->getGammaDofID(i,k)] = interior->getLocalDofID(i,k);
                    }
                }

                interfacePartitionOfUnity->computePartitionOfUnity(nodeList);
                PartitionOfUnity_ = interfacePartitionOfUnity;
            }

            // Construct local Interface nullspace basis (in case the interface was empty before, it was replaced by the interior. Therfore, this should be correct as well)
            XMapPtr serialInterfaceMap = MapFactory<LO,GO,NO>::Build(nullSpaceBasis->getMap()->lib(),this->GammaDofs_[blockId].size(),this->GammaDofs_[blockId].size(),0,this->SerialComm_);
            XMultiVectorPtr interfaceNullspaceBasis = MultiVectorFactory<SC,LO,GO,NO>::Build(serialInterfaceMap,nullSpaceBasis->getNumVectors());
            for (UN i=0; i<nullSpaceBasis->getNumVectors(); i++) {
                for (UN k=0; k<this->DofsPerNode_[blockId]; k++) {
                    for (UN j=0; j<interface->getNumNodes(); j++) {
                        interfaceNullspaceBasis->getDataNonConst(i)[interface->getGammaDofID(j,k)] = nullSpaceBasis->getData(i)[nullSpaceBasis->getMap()->getLocalElement(interface->getGlobalDofID(j,k))];
                    }
                }
            }

           PartitionOfUnity_->assembledPartitionOfUnityMaps();
           this->kRowMap_ = PartitionOfUnity_->getAssembledPartitionOfUnityMap();


            if (this->ParameterList_->get("Use RepMap",false)) {
                if (this->K_->getMap()->lib() == Xpetra::UseTpetra) {
                    //Teuchos::RCP<DDInterface<SC,LO,GO,NO> > theInterface =Teuchos::rcp_const_cast<DDInterface<SC,LO,GO,NO> >(interfacePartitionOfUnity->getDDInterface());
                    //this->buildGlobalGraph(theInterface);
                    interfacePartitionOfUnity->buildGlobalGraph();

                    Teuchos::RCP<DDInterface<SC,LO,GO,NO> > theInterface =Teuchos::rcp_const_cast<DDInterface<SC,LO,GO,NO> >(interfacePartitionOfUnity->getDDInterface());
                    this->buildGlobalGraph(theInterface);

                    this->numEnt = interfacePartitionOfUnity->getDDInterface()->getNumEnt();
                    }
                }




            // Build local basis
            LocalPartitionOfUnityBasis_ = LocalPartitionOfUnityBasisPtr(new LocalPartitionOfUnityBasis<SC,LO,GO,NO>(this->MpiComm_,this->SerialComm_,this->DofsPerNode_[blockId],sublist(coarseSpaceList,"LocalPartitionOfUnityBasis"),interfaceNullspaceBasis.getConst(),PartitionOfUnity_->getLocalPartitionOfUnity(),PartitionOfUnity_->getPartitionOfUnityMaps())); // sublist(coarseSpaceList,"LocalPartitionOfUnityBasis") testen

            LocalPartitionOfUnityBasis_->buildLocalPartitionOfUnityBasis();
            if(sublist(coarseSpaceList,"LocalPartitionOfUnityBasis")->get("Coarse NullSpace",false)){

              this->CoarseNullSpace_[blockId] = LocalPartitionOfUnityBasis_->getCoarseNullSpace();

            }

            this->InterfaceCoarseSpaces_[blockId] = LocalPartitionOfUnityBasis_->getLocalPartitionOfUnitySpace();

            if (this->Verbose_) std::cout << "FROSch::IPOUHarmonicCoarseOperator : WARNING: Need to build block coarse sizes for use in MueLu nullspace." << std::endl;
            //if (this->Verbose_) {RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(std::cout)); this->MVPhiGamma_[blockId]->describe(*fancy,VERB_EXTREME);}
        }
        return 0;
    }

  template <class SC,class LO,class GO,class NO>
  int IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::buildElementNodeList(){

  FROSCH_TIMER_START_LEVELID(buildElementNodeListTime,"CoarseOperator::buildElementNodeList");

  Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

  int MLgatheringSteps = this->DistributionList_->get("MLGatheringSteps",2);

  Teuchos::ArrayView<const GO> elements_ = this->kRowMap_->getNodeElementList();
  UN maxNumElements = -1;
  UN numElementsLocal = elements_.size();
  {
    reduceAll(*this->MpiComm_,Teuchos::REDUCE_MAX,numElementsLocal,Teuchos::ptr(&maxNumElements));
  }

  GraphPtr ElemGraph = Xpetra::CrsGraphFactory<LO,GO,NO>::Build(this->MLGatheringMaps_[0],maxNumElements);
  Teuchos::ArrayView<const GO> myGlobals = this->SubdomainConnectGraph_->getRowMap()->getNodeElementList();

  {
      Teuchos::Array<GO> col_vec(elements_.size());
      for(int i = 0; i<elements_.size(); i++) {
          col_vec.at(i) = i;
      }
      for (size_t i = 0; i < this->SubdomainConnectGraph_->getRowMap()->getNodeNumElements(); i++) {

          ElemGraph->insertGlobalIndices(myGlobals[i],elements_);
      }
      ElemGraph->fillComplete();
    }
    GraphPtr tmpElemGraph = Xpetra::CrsGraphFactory<LO,GO,NO>::Build(this->MLGatheringMaps_[1],maxNumElements);
    GraphPtr ElemSGraph;

    tmpElemGraph->doExport(*ElemGraph,*this->MLCoarseSolveExporters_[1],Xpetra::INSERT);
    UN gathered = 0;
    for(int i  = 2;i<this->MLGatheringMaps_.size();i++){
        gathered = 1;
        tmpElemGraph->fillComplete();
        ElemSGraph = tmpElemGraph;
        tmpElemGraph = Xpetra::CrsGraphFactory<LO,GO,NO>::Build(this->MLGatheringMaps_[i],maxNumElements);
        tmpElemGraph->doExport(*ElemSGraph,*this->MLCoarseSolveExporters_[i],Xpetra::INSERT);
      }

   if(gathered == 0){
     ElemSGraph = tmpElemGraph;
   }
   this->MLCoarseMap_ = MapFactory<LO,GO,NO>::Build(this->MLGatheringMaps_[1]->lib(),-1,this->MLGatheringMaps_[1]->getNodeElementList(),0,this->CoarseSolveComm_);
   this->ElementNodeList_ =Xpetra::CrsGraphFactory<LO,GO,NO>::Build(this->MLCoarseMap_,maxNumElements);

      if(this->OnCoarseSolveComm_){
          const size_t numMyElementS = this->MLCoarseMap_->getNodeNumElements();
          //Teuchos::ArrayView<const GO> myGlobalElements = MLCoarseMap_->getNodeElementList();
          //Teuchos::ArrayView<const LO> idEl;
          Teuchos::ArrayView<const GO> va;
          for (UN i = 0; i < numMyElementS; i++) {
              GO kg = this->MLGatheringMaps_[this->MLGatheringMaps_.size()-1]->getGlobalElement(i);
              ElemSGraph->getGlobalRowView(kg,va);
              Teuchos::Array<GO> vva(va);
              this->ElementNodeList_->insertGlobalIndices(kg,vva());//mal va nehmen

          }
          this->ElementNodeList_->fillComplete();
      }
      return 0;
  }

  template <class SC,class LO,class GO,class NO>
  int IPOUHarmonicCoarseOperator<SC,LO,GO,NO>::buildCoarseGraph(){
    FROSCH_TIMER_START_LEVELID(buildCoarseGraphTime,"CoarseOperator::buildCoarseGraph");

     Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

		 int nSubs = this->MpiComm_->getSize();
		 GraphPtr TestGraph2 =  Xpetra::CrsGraphFactory<LO,GO,NO>::Build(this->MLGatheringMaps_[1],this->maxNumNeigh_);;
	 	 GraphPtr TestGraph3;
		 int ee = 0;
		 {
	 	   TestGraph2->doExport(*this->SubdomainConnectGraph_,*this->MLCoarseSolveExporters_[1],Xpetra::INSERT);
		 }

	 	for(int i  = 2;i<this->MLGatheringMaps_.size();i++){
			{
			TestGraph2->fillComplete();
			TestGraph3 = TestGraph2;
	 		TestGraph2 = Xpetra::CrsGraphFactory<LO,GO,NO>::Build(this->MLGatheringMaps_[i],this->maxNumNeigh_);
	 		TestGraph2->doExport(*TestGraph3,*this->MLCoarseSolveExporters_[i],Xpetra::INSERT);
		  }
	 	}

		const size_t numMyElementS = this->MLGatheringMaps_[this->MLGatheringMaps_.size()-1]->getNodeNumElements();
    if (this->OnCoarseSolveComm_) {
			 this->SubdomainConnectGraph_= Xpetra::CrsGraphFactory<LO,GO,NO>::Build(this->MLCoarseMap_,this->maxNumNeigh_);
			 for (size_t k = 0; k<numMyElementS; k++) {
				 Teuchos::ArrayView<const LO> in;
         Teuchos::ArrayView<const GO> vals_graph;
         GO kg = this->MLGatheringMaps_[this->MLGatheringMaps_.size()-1]->getGlobalElement(k);
         TestGraph2->getGlobalRowView(kg,vals_graph);
         Teuchos::Array<GO> vals(vals_graph);
				 this->SubdomainConnectGraph_->insertGlobalIndices(kg,vals());
			 }
			 this->SubdomainConnectGraph_->fillComplete();
		 }
        return 0;
  }

}

#endif
