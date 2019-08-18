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

#ifndef _FROSCH_COARSEOPERATOR_DECL_HPP
#define _FROSCH_COARSEOPERATOR_DECL_HPP

#include <FROSch_SchwarzOperator_def.hpp>
#include <Teuchos_TimeMonitor.hpp>
#define FROSch_CoarseOperatorTimer

// TODO: Member sortieren!?


namespace FROSch {

    template <class SC = Xpetra::Operator<>::scalar_type,
              class LO = typename Xpetra::Operator<SC>::local_ordinal_type,
              class GO = typename Xpetra::Operator<SC,LO>::global_ordinal_type,
              class NO = typename Xpetra::Operator<SC,LO,GO>::node_type>
    class CoarseOperator : public SchwarzOperator<SC,LO,GO,NO> {

    protected:

        using CommPtr               = typename SchwarzOperator<SC,LO,GO,NO>::CommPtr;

        using Map                   = typename SchwarzOperator<SC,LO,GO,NO>::Map;
        using MapPtr                = typename SchwarzOperator<SC,LO,GO,NO>::MapPtr;
        using ConstMapPtr           = typename SchwarzOperator<SC,LO,GO,NO>::ConstMapPtr;
        using MapPtrVecPtr          = typename SchwarzOperator<SC,LO,GO,NO>::MapPtrVecPtr;
        using ConstMapPtrVecPtr     = typename SchwarzOperator<SC,LO,GO,NO>::ConstMapPtrVecPtr;

        using CrsMatrixPtr          = typename SchwarzOperator<SC,LO,GO,NO>::CrsMatrixPtr;
        using ConstCrsMatrixPtr     = typename SchwarzOperator<SC,LO,GO,NO>::ConstCrsMatrixPtr;

        using MultiVector           = typename SchwarzOperator<SC,LO,GO,NO>::MultiVector;
        using MultiVectorPtr        = typename SchwarzOperator<SC,LO,GO,NO>::MultiVectorPtr;

        using ExporterPtrVecPtr     = typename SchwarzOperator<SC,LO,GO,NO>::ExporterPtrVecPtr;

        using ParameterList         = typename SchwarzOperator<SC,LO,GO,NO>::ParameterList;
        using ParameterListPtr      = typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr;

        using CoarseSpacePtr        = typename SchwarzOperator<SC,LO,GO,NO>::CoarseSpacePtr;

        using SubdomainSolverPtr    = typename SchwarzOperator<SC,LO,GO,NO>::SubdomainSolverPtr;

        using UN                    = typename SchwarzOperator<SC,LO,GO,NO>::UN;

        using GOVec                 = typename SchwarzOperator<SC,LO,GO,NO>::GOVec;
        using GOVecPtr              = typename SchwarzOperator<SC,LO,GO,NO>::GOVecPtr;
        using GOVec2D               = Teuchos::Array<GOVec>;

        using LOVec                 = typename SchwarzOperator<SC,LO,GO,NO>::LOVec;
        using LOVecPtr2D            = typename SchwarzOperator<SC,LO,GO,NO>::LOVecPtr2D;

        using IntVec                = typename SchwarzOperator<SC,LO,GO,NO>::IntVec;
        using IntVec2D              = Teuchos::Array<IntVec>;

        using SCVec                 = typename SchwarzOperator<SC,LO,GO,NO>::SCVec;

        using ConstLOVecView        = typename SchwarzOperator<SC,LO,GO,NO>::ConstLOVecView;

        using ConstGOVecView        = typename SchwarzOperator<SC,LO,GO,NO>::ConstGOVecView;

        using ConstSCVecView        = typename SchwarzOperator<SC,LO,GO,NO>::ConstSCVecView;

        using TimePtr               = typename SchwarzOperator<SC,LO,GO,NO>::TimePtr;


        using GraphPtr             = typename SchwarzOperator<SC,LO,GO,NO>::GraphPtr;
        using EntitySetPtr            = typename SchwarzOperator<SC,LO,GO,NO>::EntitySetPtr;
        using EntitySetConstPtr       = const EntitySetPtr;
        using EntitySetPtrVecPtr      = Teuchos::ArrayRCP<EntitySetPtr>;
        using EntitySetPtrConstVecPtr =  const EntitySetPtrVecPtr;


        using InterfaceEntityPtr        = Teuchos::RCP<InterfaceEntity<SC,LO,GO,NO> >;
        using InterfaceEntityPtrVec     = Teuchos::Array<InterfaceEntityPtr>;
        using InterfaceEntityPtrVecPtr  = Teuchos::ArrayRCP<InterfaceEntityPtr>;


    public:

        CoarseOperator(ConstCrsMatrixPtr k,
                       ParameterListPtr parameterList);

        ~CoarseOperator();

        virtual int initialize() = 0;

        virtual int compute();

        virtual MapPtr computeCoarseSpace(CoarseSpacePtr coarseSpace) = 0;

        virtual int clearCoarseSpace();

        virtual void apply(const MultiVector &x,
                          MultiVector &y,
                          bool usePreconditionerOnly,
                          Teuchos::ETransp mode=Teuchos::NO_TRANS,
                          SC alpha=Teuchos::ScalarTraits<SC>::one(),
                          SC beta=Teuchos::ScalarTraits<SC>::zero()) const;

        virtual void applyPhiT(MultiVector& x,
                              MultiVector& y) const;

        virtual void applyCoarseSolve(MultiVector& x,
                                     MultiVector& y,
                                     Teuchos::ETransp mode=Teuchos::NO_TRANS) const;

        virtual void applyPhi(MultiVector& x,
                             MultiVector& y) const;

        virtual CoarseSpacePtr getCoarseSpace() const;


        static int current_level;

    protected:

        virtual MapPtr assembleSubdomainMap() = 0;

        virtual int setUpCoarseOperator();

        CrsMatrixPtr buildCoarseMatrix();

        virtual int buildCoarseSolveMap(CrsMatrixPtr &k0);
//functions for coarse RepeatedMap
        virtual int buildElementNodeList();
        virtual int buildGlobalGraph(Teuchos::RCP<DDInterface<SC,LO,GO,NO> > theDDInterface_);
        virtual int buildCoarseGraph();
//

        CommPtr CoarseSolveComm_;

        bool OnCoarseSolveComm_;

        LO NumProcsCoarseSolve_;

        CoarseSpacePtr CoarseSpace_;

        CrsMatrixPtr Phi_;
        CrsMatrixPtr CoarseMatrix_;

        ConstMapPtrVecPtr GatheringMaps_;
         MapPtrVecPtr MLGatheringMaps_;
        MapPtr CoarseSolveMap_;
        MapPtr CoarseSolveRepeatedMap_;
        MapPtr MLCoarseMap_;

        SubdomainSolverPtr CoarseSolver_;

        ParameterListPtr DistributionList_;

        ExporterPtrVecPtr CoarseSolveExporters_;
        ExporterPtrVecPtr MLCoarseSolveExporters_;
        //Graph to compute Reapeated Map
        GraphPtr SubdomainConnectGraph_;
        //Element-Node-List to compute RepeatedMap
        GraphPtr ElementNodeList_;
        Teuchos::RCP<Xpetra::CrsMatrix<GO,LO,GO,NO> > GraphEntriesList_;

        ConstMapPtr kRowMap_;
        LO DofsPerNodeCoarse_;
        UN dofs;
        UN maxNumNeigh_;

    #ifdef FROSch_CoarseOperatorTimer
	   Teuchos::Array<TimePtr> ComputeTimer;
	   Teuchos::Array<TimePtr> ApplyTimer;
	   Teuchos::Array<TimePtr> ApplyPhiTTimer;
	   Teuchos::Array<TimePtr> ApplyExportTimer;
	   Teuchos::Array<TimePtr> ApplyCoarseSolveTimer;
	   Teuchos::Array<TimePtr> ApplyPhiTimer;
	   Teuchos::Array<TimePtr> ApplyImportTimer;
	   Teuchos::Array<TimePtr> SetUpTimer;
	   Teuchos::Array<TimePtr> BuildCoarseMatrixTimer;
	   Teuchos::Array<TimePtr> BuildCoarseSolveMapTimer;
	   Teuchos::Array<TimePtr> BuildCoarseRepMapTimer;
	   Teuchos::Array<TimePtr> ExportKOTimer;
	   Teuchos::Array<TimePtr> BuildDirectSolvesTimer;
     Teuchos::Array<TimePtr> BuildGlobalGraphTimer;
     Teuchos::Array<TimePtr> InterfaceInfoTimer;
     Teuchos::Array<TimePtr> BuildCoarseGraphTimer;
     Teuchos::Array<TimePtr> BuildElementNodeListTimer;
     Teuchos::Array<TimePtr> CompAssembleCoarseSpaceTimer;
     Teuchos::Array<TimePtr> CompBuildBasisMatrixTimer;
     Teuchos::Array<TimePtr> CompCoarseSpaceTimer;
     Teuchos::Array<TimePtr> ExportCMatrixTimer;
      #endif

    };

}

#endif
