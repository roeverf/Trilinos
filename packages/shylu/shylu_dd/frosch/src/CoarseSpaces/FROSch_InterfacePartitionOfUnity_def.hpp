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

#ifndef _FROSCH_INTERFACEPARTITIONOFUNITY_DEF_HPP
#define _FROSCH_INTERFACEPARTITIONOFUNITY_DEF_HPP

#include <FROSch_InterfacePartitionOfUnity_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    InterfacePartitionOfUnity<SC,LO,GO,NO>::InterfacePartitionOfUnity(CommPtr mpiComm,
                                                                      CommPtr serialComm,
                                                                      UN dimension,
                                                                      UN dofsPerNode,
                                                                      ConstXMapPtr nodesMap,
                                                                      ConstXMapPtrVecPtr dofsMaps,
                                                                      ParameterListPtr parameterList,
                                                                      Verbosity verbosity,
                                                                      UN levelID,
                                                                      UN NumLevel) :
    PartitionOfUnity<SC,LO,GO,NO> (mpiComm,serialComm,dofsPerNode,nodesMap,dofsMaps,parameterList,verbosity,levelID,NumLevel),
    DDInterface_ ()
    {
        FROSCH_TIMER_START_LEVELID(interfacePartitionOfUnityTime,"InterfacePartitionOfUnity::InterfacePartitionOfUnity");
        CommunicationStrategy communicationStrategy = CreateOneToOneMap;
        if (!this->ParameterList_->get("Interface Communication Strategy","CreateOneToOneMap").compare("CrsMatrix")) {
            communicationStrategy = CommCrsMatrix;
        } else if (!this->ParameterList_->get("Interface Communication Strategy","CreateOneToOneMap").compare("CrsGraph")) {
            communicationStrategy = CommCrsGraph;
        } else if (!this->ParameterList_->get("Interface Communication Strategy","CreateOneToOneMap").compare("CreateOneToOneMap")) {
            communicationStrategy = CreateOneToOneMap;
        } else {
            FROSCH_ASSERT(false,"FROSch::InterfacePartitionOfUnity : ERROR: Specify a valid communication strategy for the identification of the interface components.");
        }

        DDInterface_.reset(new DDInterface<SC,LO,GO,NO>(dimension,dofsPerNode,nodesMap.getConst(),this->Verbosity_,this->LevelID_,communicationStrategy));
        DDInterface_->resetGlobalDofs(dofsMaps);
    }

    template <class SC,class LO,class GO,class NO>
    InterfacePartitionOfUnity<SC,LO,GO,NO>::~InterfacePartitionOfUnity()
    {

    }

    template <class SC,class LO,class GO,class NO>
    typename InterfacePartitionOfUnity<SC,LO,GO,NO>::ConstDDInterfacePtr InterfacePartitionOfUnity<SC,LO,GO,NO>::getDDInterface() const
    {
        return DDInterface_.getConst();
    }

    template <class SC,class LO,class GO,class NO>
    typename InterfacePartitionOfUnity<SC,LO,GO,NO>::DDInterfacePtr InterfacePartitionOfUnity<SC,LO,GO,NO>::getDDInterfaceNonConst() const
    {
        return DDInterface_;
    }

    template<class SC,class LO, class GO, class NO>
    int InterfacePartitionOfUnity<SC,LO,GO,NO>::buildGlobalGraph(){


     FROSCH_TIMER_START_LEVELID(buildGlobalGraphTime,"CoarseOperator::buildGlobalGraph");

     Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
     std::map<GO,int> rep;
     Teuchos::Array<GO> entries;
     IntVec2D conn;
     InterfaceEntityPtrVec ConnVec;
     int connrank;
     {
       DDInterface_->identifyConnectivityEntities();
       EntitySetConstPtr Connect= DDInterface_->getConnectivityEntities();
       Connect->buildEntityMap(DDInterface_->getNodesMap());
       ConnVec = Connect->getEntityVector();
       connrank = Connect->getEntityMap()->getComm()->getRank();
     }

     GO ConnVecSize = ConnVec.size();
     conn.resize(ConnVecSize);
     {
       if (ConnVecSize>0) {
         for(GO i = 0;i<ConnVecSize;i++) {
             conn[i] = ConnVec[i]->getSubdomainsVector();
             for (int j = 0; j<conn[i].size(); j++) rep.insert(std::pair<GO,int>(conn.at(i).at(j),connrank));
         }
         for (auto& x: rep) {
             entries.push_back(x.first);
         }
       }
     }

     Teuchos::RCP<Xpetra::Map<LO,GO,NO> > GraphMap = Xpetra::MapFactory<LO,GO,NO>::Build(Xpetra::UseTpetra,-1,1,0,this->MpiComm_);

     UN maxNumElements = -1;
     maxNumNeigh_ = -1;
     UN numElementsLocal = entries.size();
     reduceAll(*this->MpiComm_,Teuchos::REDUCE_MAX,numElementsLocal,Teuchos::ptr(&maxNumNeigh_));
     SubdomainConnectGraph_ = Xpetra::CrsGraphFactory<LO,GO,NO>::Build(GraphMap,maxNumNeigh_);
     SubdomainConnectGraph_->insertGlobalIndices(GraphMap->getComm()->getRank(),entries());

     return 0;
   }

    template<class SC,class LO, class GO, class NO>
    typename InterfacePartitionOfUnity<SC,LO,GO,NO>::GraphPtr InterfacePartitionOfUnity<SC,LO,GO,NO>::getSubdomainGraph() const
    {
      return SubdomainConnectGraph_;
    }
}

#endif
