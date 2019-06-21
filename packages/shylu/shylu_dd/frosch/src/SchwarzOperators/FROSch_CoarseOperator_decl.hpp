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
//#define FROSch_CoarseOperatorTimers
// TODO: Member sortieren!?


namespace FROSch {

    template <class SC = Xpetra::Operator<>::scalar_type,
    class LO = typename Xpetra::Operator<SC>::local_ordinal_type,
    class GO = typename Xpetra::Operator<SC,LO>::global_ordinal_type,
    class NO = typename Xpetra::Operator<SC,LO,GO>::node_type>
    class CoarseOperator : public SchwarzOperator<SC,LO,GO,NO> {

    public:

        typedef typename SchwarzOperator<SC,LO,GO,NO>::CommPtr CommPtr;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::Map Map;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::MapPtr MapPtr;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::MapPtrVecPtr MapPtrVecPtr;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::ConstMapPtr ConstMapPtr;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::CrsMatrixPtr CrsMatrixPtr;
        typedef typename Teuchos::RCP<Xpetra::CrsMatrix<GO,LO,GO,NO> >GOCrsMatrixPtr;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::MultiVector MultiVector;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::MultiVectorPtr MultiVectorPtr;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::ExporterPtrVecPtr ExporterPtrVecPtr;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::ParameterList ParameterList;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr ParameterListPtr;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::CoarseSpacePtr CoarseSpacePtr;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::SubdomainSolverPtr SubdomainSolverPtr;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::UN UN;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::GOVec GOVec;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::GOVecPtr GOVecPtr;
        typedef Teuchos::Array<GOVec> GOVec2D;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::LOVec LOVec;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::LOVecPtr2D LOVecPtr2D;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::SCVec SCVec;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::ConstLOVecView ConstLOVecView;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::ConstGOVecView ConstGOVecView;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::ConstSCVecView ConstSCVecView;
        typedef typename SchwarzOperator<SC,LO,GO,NO>::CrsGraphPtr CrsGraphPtr;

        typedef typename SchwarzOperator<SC,LO,GO,NO>::EntitySetPtr EntitySetPtr;
        typedef const EntitySetPtr EntitySetConstPtr;
        typedef Teuchos::ArrayRCP<EntitySetPtr> EntitySetPtrVecPtr;
        typedef const EntitySetPtrVecPtr EntitySetPtrConstVecPtr;


        typedef Teuchos::RCP<InterfaceEntity<SC,LO,GO,NO> > InterfaceEntityPtr;
        typedef Teuchos::Array<InterfaceEntityPtr> InterfaceEntityPtrVec;
        typedef Teuchos::ArrayRCP<InterfaceEntityPtr> InterfaceEntityPtrVecPtr;

		typedef typename SchwarzOperator<SC,LO,GO,NO>::Time Time;
		typedef typename SchwarzOperator<SC,LO,GO,NO>::TimePtr TimePtr;



        CoarseOperator(CrsMatrixPtr k,
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
        virtual int buildElementNodeList();
        virtual int buildGraphEntries(Teuchos::RCP<DDInterface<SC,LO,GO,NO> > theDDInterface_);
        virtual int buildConnectGraph();
        CommPtr CoarseSolveComm_;

        bool OnCoarseSolveComm_;

        LO NumProcsCoarseSolve_;

        CoarseSpacePtr CoarseSpace_;

        CrsMatrixPtr Phi_;
        CrsMatrixPtr CoarseMatrix_;

        MapPtrVecPtr GatheringMaps_;
        MapPtrVecPtr MLGatheringMaps_;
        MapPtr CoarseSolveMap_;
        MapPtr CoarseSolveRepeatedMap_;
        GOVecPtr BlockCoarseDimension_;

        SubdomainSolverPtr CoarseSolver_;
        ParameterListPtr DistributionList_;

        ExporterPtrVecPtr CoarseSolveExporters_;
        ExporterPtrVecPtr MLCoarseSolveExporters_;
        //Graph to compute Reapeated Map
        CrsGraphPtr SubdomainConnectGraph_;
        //Element-Node-List to compute RepeatedMap
        Teuchos::RCP<Xpetra::CrsMatrix<GO,LO,GO,NO> > ElementNodeList_;
        Teuchos::RCP<Xpetra::CrsMatrix<GO,LO,GO,NO> > GraphEntriesList_;

        ConstMapPtr kRowMap_;
        LO DofsPerNodeCoarse_;
        UN dofs;

#ifdef FROSch_CoarseOperatorTimers
	 Teuchos::Array<TimePtr> ComputeTimer;
	Teuchos::Array<TimePtr> BuildGraphEntriesTimer;
	Teuchos::Array<TimePtr> BuildGraphTimer;
	Teuchos::Array<TimePtr> GraphTimer1;

	Teuchos::Array<TimePtr> BuildElementNodeListTimer;

	Teuchos::Array<TimePtr>ElementNodeListTimer1;
	Teuchos::Array<TimePtr>ElementNodeListTimer2;
	Teuchos::Array<TimePtr>ElementNodeListTimer2;
	Teuchos::Array<TimePtr>ElementNodeListTimer3;
	Teuchos::Array<TimePtr>ElementNodeListTimer4;

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
#endif
    };

}

#endif
