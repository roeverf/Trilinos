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
// Questions? Contact Christian Hochmuth (c.hochmuth@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_TWOLEVELBLOCKPRECONDITIONER_DECL_HPP
#define _FROSCH_TWOLEVELBLOCKPRECONDITIONER_DECL_HPP

#include <FROSch_OneLevelPreconditioner_def.hpp>

namespace FROSch {

    template <class SC = Xpetra::Operator<>::scalar_type,
              class LO = typename Xpetra::Operator<SC>::local_ordinal_type,
              class GO = typename Xpetra::Operator<SC, LO>::global_ordinal_type,
              class NO = typename Xpetra::Operator<SC, LO, GO>::node_type>
    class TwoLevelBlockPreconditioner : public OneLevelPreconditioner<SC,LO,GO,NO> {

    protected:

        using MapPtr                              = typename SchwarzPreconditioner<SC,LO,GO,NO>::MapPtr;
        using ConstMapPtr                         = typename SchwarzPreconditioner<SC,LO,GO,NO>::ConstMapPtr;
        using MapPtrVecPtr                        = typename SchwarzPreconditioner<SC,LO,GO,NO>::MapPtrVecPtr;
        using ConstMapPtrVecPtr                   = typename SchwarzPreconditioner<SC,LO,GO,NO>::ConstMapPtrVecPtr;
        using MapPtrVecPtr2D                      = typename SchwarzPreconditioner<SC,LO,GO,NO>::MapPtrVecPtr2D;
        using ConstMapPtrVecPtr2D                 = typename SchwarzPreconditioner<SC,LO,GO,NO>::ConstMapPtrVecPtr2D;

        using CrsMatrixPtr                        = typename SchwarzPreconditioner<SC,LO,GO,NO>::CrsMatrixPtr;
        using ConstCrsMatrixPtr                   = typename SchwarzPreconditioner<SC,LO,GO,NO>::ConstCrsMatrixPtr;

        using MultiVectorPtr                      = typename SchwarzPreconditioner<SC,LO,GO,NO>::MultiVectorPtr;
        using MultiVectorPtrVecPtr                = typename SchwarzPreconditioner<SC,LO,GO,NO>::MultiVectorPtrVecPtr;
        using ConstMultiVectorPtrVecPtr           = typename SchwarzPreconditioner<SC,LO,GO,NO>::ConstMultiVectorPtrVecPtr;

        using ParameterListPtr                    = typename SchwarzPreconditioner<SC,LO,GO,NO>::ParameterListPtr;

        using AlgebraicOverlappingOperatorPtr     = typename SchwarzPreconditioner<SC,LO,GO,NO>::AlgebraicOverlappingOperatorPtr;
        using CoarseOperatorPtr                   = typename SchwarzPreconditioner<SC,LO,GO,NO>::CoarseOperatorPtr;
        using GDSWCoarseOperatorPtr               = typename SchwarzPreconditioner<SC,LO,GO,NO>::GDSWCoarseOperatorPtr;
        using RGDSWCoarseOperatorPtr              = typename SchwarzPreconditioner<SC,LO,GO,NO>::RGDSWCoarseOperatorPtr;
        using IPOUHarmonicCoarseOperatorPtr       = typename SchwarzPreconditioner<SC,LO,GO,NO>::IPOUHarmonicCoarseOperatorPtr;

        using UN                                  = typename SchwarzPreconditioner<SC,LO,GO,NO>::UN;

        using GOVec                               = typename SchwarzPreconditioner<SC,LO,GO,NO>::GOVec;
        using GOVec2D                             = typename SchwarzPreconditioner<SC,LO,GO,NO>::GOVec2D;
        using GOVecPtr                            = typename SchwarzPreconditioner<SC,LO,GO,NO>::GOVecPtr;
        using UNVecPtr                            = typename SchwarzPreconditioner<SC,LO,GO,NO>::UNVecPtr;
        using LOVecPtr                            = typename SchwarzPreconditioner<SC,LO,GO,NO>::LOVecPtr;
        using GOVecPtr2D                          = typename SchwarzPreconditioner<SC,LO,GO,NO>::GOVecPtr2D;
        using DofOrderingVecPtr                   = typename Teuchos::ArrayRCP<DofOrdering>;

    public:

        TwoLevelBlockPreconditioner(ConstCrsMatrixPtr k,
                                    ParameterListPtr parameterList);

        int initialize(UN dimension,
                       UNVecPtr dofsPerNodeVec,
                       DofOrderingVecPtr dofOrderingVec,
                       int overlap = -1,
                       ConstMapPtrVecPtr repeatedMapVec = Teuchos::null,
                       ConstMultiVectorPtrVecPtr nullSpaceBasisVec = Teuchos::null,
                       ConstMultiVectorPtrVecPtr nodeListVec = Teuchos::null,
                       ConstMapPtrVecPtr2D dofsMapsVec = Teuchos::null,
                       GOVecPtr2D dirichletBoundaryDofsVec = Teuchos::null);

        int compute();

        void describe(Teuchos::FancyOStream &out,
                      const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const;

        std::string description() const;

        int resetMatrix(ConstCrsMatrixPtr &k);

        int preApplyCoarse(MultiVectorPtr &x,MultiVectorPtr &y);

    protected:

        CoarseOperatorPtr CoarseOperator_;

    };

}

#endif
