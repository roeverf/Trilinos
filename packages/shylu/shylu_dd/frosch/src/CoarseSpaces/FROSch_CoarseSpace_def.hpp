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

#ifndef _FROSCH_COARSESPACE_DEF_HPP
#define _FROSCH_COARSESPACE_DEF_HPP

#include <FROSch_CoarseSpace_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    CoarseSpace<SC,LO,GO,NO>::CoarseSpace(CommPtr mpiComm,
                                          CommPtr serialComm) :
    MpiComm_ (mpiComm),
    SerialComm_ (serialComm)
    {

    }

    // Will man Informationen über die Subspaces als strings reingeben?
    template <class SC,class LO,class GO,class NO>
    int CoarseSpace<SC,LO,GO,NO>::addSubspace(ConstXMapPtr subspaceBasisMap,
                                              ConstXMapPtr subspaceBasisMapUnique,
                                              ConstXMultiVectorPtr subspaceBasis,
                                              UN offset)
    {
        FROSCH_ASSERT(!subspaceBasisMap.is_null(),"FROSch::CoarseSpace : ERROR: subspaceBasisMap.is_null()");
        if (!subspaceBasis.is_null()) {
            FROSCH_ASSERT(subspaceBasis->getNumVectors()==subspaceBasisMap->getNodeNumElements(),"FROSch::CoarseSpace : ERROR: subspaceBasis->getNumVectors()!=subspaceBasisMap->getNodeNumElements()");
        } else {
            FROSCH_ASSERT(subspaceBasisMap->getNodeNumElements()==0,"FROSch::CoarseSpace : ERROR: subspaceBasisMap->getNodeNumElements()!=0");
        }

        UnassembledBasesMaps_.push_back(subspaceBasisMap);
        UnassembledBasesMapsUnique_.push_back(subspaceBasisMapUnique);
        UnassembledSubspaceBases_.push_back(subspaceBasis);
        Offsets_.push_back(offset);
        LocalSubspacesSizes_.push_back(subspaceBasisMap->getNodeNumElements());

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int CoarseSpace<SC,LO,GO,NO>::addNullspace(ConstXMapPtr subspaceBasisMap,ConstXMultiVectorPtr nullSpaceBasis)
    {
        UnassembledNullSpaceMaps_.push_back(subspaceBasisMap);
        UnassembledNullSpaceBases_.push_back(nullSpaceBasis);
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int CoarseSpace<SC,LO,GO,NO>::assembleCoarseSpace()
    {
        FROSCH_ASSERT(UnassembledBasesMaps_.size()>0,"FROSch::CoarseSpace : ERROR: UnassembledBasesMaps_.size()==0");
        FROSCH_ASSERT(UnassembledBasesMapsUnique_.size()>0,"FROSch::CoarseSpace : ERROR: UnassembledBasesMapsUnique_.size()==0");
        FROSCH_ASSERT(UnassembledSubspaceBases_.size()>0,"FROSch::CoarseSpace : ERROR: UnassembledSubspaceBases_.size()==0");

        UN itmp = 0;
        LOVecPtr2D partMappings;

        // BasisMap
        AssembledBasisMap_ = AssembleMaps(UnassembledBasesMaps_(),partMappings);

        // BasisMapUnique - First, we check if any of the unassembled unique maps is null. In case, we re-build a unique map
        bool buildUniqueMap = false;
        UN i=0;
        while (!buildUniqueMap && i<UnassembledBasesMapsUnique_.size()) {
            buildUniqueMap = UnassembledBasesMapsUnique_[i].is_null();
            i++;
        }
        int buildUniqueMapMax = 0;
        reduceAll(*this->MpiComm_,REDUCE_MAX,int(buildUniqueMap),ptr(&buildUniqueMapMax));

        if (buildUniqueMapMax>0) {
            FROSCH_NOTIFICATION("FROSch::CoarseSpace",this->MpiComm_->getRank()==0,"We re-build a unique map of AssembledBasisMap_.");
            AssembledBasisMapUnique_ = BuildUniqueMap<LO,GO,NO>(AssembledBasisMap_);
        } else {
            AssembledBasisMapUnique_ = AssembleMaps(UnassembledBasesMapsUnique_(),partMappings);
        }
        FROSCH_ASSERT(AssembledBasisMap_->getMaxAllGlobalIndex()==AssembledBasisMapUnique_->getMaxAllGlobalIndex(),"FROSch::CoarseSpace : ERROR: AssembledBasisMap_->getMaxAllGlobalIndex()!=AssembledBasisMapUnique_->getMaxAllGlobalIndex()");
        FROSCH_ASSERT(AssembledBasisMap_->getMinAllGlobalIndex()==AssembledBasisMapUnique_->getMinAllGlobalIndex(),"FROSch::CoarseSpace : ERROR: AssembledBasisMap_->getMinAllGlobalIndex()!=AssembledBasisMapUnique_->getMinAllGlobalIndex()");
        FROSCH_ASSERT(GO(AssembledBasisMapUnique_->getGlobalNumElements())==GO(AssembledBasisMapUnique_->getMaxAllGlobalIndex()+1),"FROSch::CoarseSpace : ERROR: AssembledBasisMapUnique_->getGlobalNumElements()!=(AssembledBasisMapUnique_->getMaxAllGlobalIndex()+1)");

        // Basis
        if (!AssembledBasisMap_.is_null()) {
            if (AssembledBasisMap_->getGlobalNumElements()>0) { // AH 02/12/2019: Is this the right condition? Seems to work for now...
                LO totalSize = -1;
                for (UN i=0; i<UnassembledSubspaceBases_.size(); i++) {
                    if (!UnassembledSubspaceBases_[i].is_null()) totalSize = std::max(totalSize,LO(UnassembledSubspaceBases_[i]->getLocalLength()+Offsets_[i]));
                }
                XMapPtr serialMap = MapFactory<LO,GO,NO>::Build(AssembledBasisMap_->lib(),totalSize,0,this->SerialComm_);

                AssembledBasis_ = MultiVectorFactory<SC,LO,GO,NO >::Build(serialMap,AssembledBasisMap_->getNodeNumElements());
                for (UN i=0; i<UnassembledSubspaceBases_.size(); i++) {
                    if (!UnassembledSubspaceBases_[i].is_null()) {
                        for (UN j=0; j<UnassembledSubspaceBases_[i]->getNumVectors(); j++) {
                            ConstSCVecPtr unassembledSubspaceBasesData = UnassembledSubspaceBases_[i]->getData(j);
                            for (UN k=0; k<UnassembledSubspaceBases_[i]->getLocalLength(); k++) {
                                FROSCH_ASSERT(itmp<AssembledBasis_->getNumVectors(),"FROSch::CoarseSpace : ERROR: itmp>=AssembledBasis_->getNumVectors()");
                                FROSCH_ASSERT(k+Offsets_[i]<AssembledBasis_->getLocalLength(),"FROSch::CoarseSpace : ERROR: k+Offsets_[i]>=AssembledBasis_->getLocalLength()");
                                AssembledBasis_->replaceLocalValue(k+Offsets_[i],itmp,unassembledSubspaceBasesData[k]);
                            }
                            itmp++;
                        }
                    }
                }
            }
        }

        ConstXMapPtrVec emptyVec1;
        UnassembledBasesMaps_.swap(emptyVec1);

        ConstXMapPtrVec emptyVec2;
        UnassembledBasesMapsUnique_.swap(emptyVec2);

        ConstXMultiVectorPtrVec emtpyVec3;
        UnassembledSubspaceBases_.swap(emtpyVec3);

        LOVec emptyVec4;
        Offsets_.swap(emptyVec4);

        UnassembledBasesMaps_.push_back(AssembledBasisMap_);
        UnassembledBasesMapsUnique_.push_back(AssembledBasisMapUnique_);
        UnassembledSubspaceBases_.push_back(AssembledBasis_);
        Offsets_.push_back(0);

        return 0;
    }


    template <class SC,class LO,class GO,class NO>
    int CoarseSpace<SC,LO,GO,NO>::assembleNullSpace(UN NumRowEntries){
        FROSCH_ASSERT(UnassembledBasesMaps_.size()>0,"UnassembledBasesMaps_.size()==0");
        FROSCH_ASSERT(UnassembledSubspaceBases_.size()>0,"UnassembledSubspaceBases_.size()==0");
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::VerboseObjectBase::getDefaultOStream();

       UN itmp = 0;
       ConstXMapPtr AssembledNullSpaceMap_ = AssembledBasisMap_;

       //AssembledBasisMap_->describe(*fancy,Teuchos::VERB_EXTREME);
       XMultiVectorPtr CoarseNullSpace_ = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(AssembledNullSpaceMap_,NumRowEntries);
       if (!AssembledNullSpaceMap_.is_null()) {
          if(AssembledNullSpaceMap_->getGlobalNumElements()>0){
            for(UN i = 0; i<UnassembledNullSpaceBases_.size();i++){
              //UnassembledNullSpaceBases_[i]->describe(*fancy,Teuchos::VERB_EXTREME);
              for(UN j = 0;j<UnassembledNullSpaceBases_[i]->getNumVectors();j++){
                for(UN k = 0;k<UnassembledNullSpaceBases_[i]->getMap()->getNodeNumElements();k++){
                  CoarseNullSpace_->replaceLocalValue(itmp,k,UnassembledNullSpaceBases_[i]->getData(j)[k]);
                }
                itmp++;

              //  Teuchos::ArrayRCP<SC> vals = UnassembledNullSpaceBases_[i]->getDataNonConst(j);
                /*for(UN k = 0;k<vals.size();k++){
                  //CoarseNullSpace_->replaceGlobalValue(itmp,j,vals[k]);
                  itmp++;
                }*/
              }
            }
          }
        }
        /*XMapPtr serialMap = Xpetra::MapFactory<LO,GO,NO>::Build(AssembledNullSpaceMap_->lib(),UnassembledNullSpaceBases_[0]->getMap()->getGlobalNumElements(),UnassembledNullSpaceBases_[0]->getMap()->getGlobalNumElements(),0,UnassembledNullSpaceBases_[0]->getMap()->getComm());
        serialMap->describe(*fancy,Teuchos::VERB_EXTREME);
        if (!AssembledNullSpaceMap_.is_null()) {
          if(AssembledNullSpaceMap_->getGlobalNumElements()>0){
            AssembledNullSpace_ = Xpetra::MultiVectorFactory<SC,LO,GO,NO >::Build(serialMap,AssembledNullSpaceMap_->getNodeNumElements());
            for(UN i = 0; i<UnassembledNullSpaceMaps_.size();i++){
              for(UN j = 0;j<UnassembledNullSpaceMaps_[i]->getNodeNumElements();j++){
                 AssembledNullSpace_->getDataNonConst(itmp).deepCopy(UnassembledNullSpaceBases_[i]->getData(j)());
                 itmp++;
              }
            }
          }
        }
   */
      //CoarseNullSpace_->describe(*fancy,Teuchos::VERB_EXTREME);
      UnassembledNullSpaceMaps_.resize(0);
      AssembledNullSpace_ = CoarseNullSpace_;
      //UnassembledNullSpaceBases_.resize(0);

      UnassembledNullSpaceMaps_.push_back(AssembledNullSpaceMap_);
      //nassembledNullSpaceBases_.push_back(AssembledNullSpace_);
      return 0;


    }

    template <class SC,class LO,class GO,class NO>
    int CoarseSpace<SC,LO,GO,NO>::buildGlobalBasisMatrix(ConstXMapPtr rowMap,
                                                         ConstXMapPtr rangeMap,
                                                         ConstXMapPtr repeatedMap,
                                                         SC treshold)
    {
        FROSCH_ASSERT(!AssembledBasisMap_.is_null(),"FROSch::CoarseSpace : ERROR: AssembledBasisMap_.is_null().");
        FROSCH_ASSERT(!AssembledBasis_.is_null(),"FROSch::CoarseSpace : ERROR: AssembledBasis_.is_null().");

        GlobalBasisMatrix_ = MatrixFactory<SC,LO,GO,NO>::Build(rowMap,AssembledBasisMap_->getNodeNumElements()); // Nonzeroes abhängig von dim/dofs!!!

        LO iD;
        SC valueTmp;

        for (UN i=0; i<AssembledBasis_->getLocalLength(); i++) {
            GOVec indices;
            SCVec values;
            for (UN j=0; j<AssembledBasis_->getNumVectors(); j++) {
                valueTmp=AssembledBasis_->getData(j)[i];
                if (fabs(valueTmp)>treshold) {
                    indices.push_back(AssembledBasisMap_->getGlobalElement(j));
                    values.push_back(valueTmp);
                }
            }
            iD = rowMap->getLocalElement(repeatedMap->getGlobalElement(i));

            if (iD!=-1) {
                GlobalBasisMatrix_->insertGlobalValues(iD,indices(),values());
            }
        }
        GlobalBasisMatrix_->fillComplete(AssembledBasisMapUnique_,rangeMap);
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
   int CoarseSpace<SC,LO,GO,NO>::buildGlobalNullSpace(SC treshold)
   {
       FROSCH_ASSERT(!AssembledBasisMap_.is_null(),"AssembledBasisMap_.is_null().");
       FROSCH_ASSERT(!AssembledBasis_.is_null(),"AssembledBasis_.is_null().");
       Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::VerboseObjectBase::getDefaultOStream();

       XMapPtr rowMap = Xpetra::MapFactory<LO,GO,NO>::Build(AssembledNullSpaceMap_->lib(),AssembledNullSpace_->getMap()->getGlobalNumElements(),AssembledNullSpace_->getMap()->getGlobalNumElements(),0,AssembledNullSpaceMap_->getComm());
       //rowMap->describe(*fancy,Teuchos::VERB_EXTREME);
       /*GlobalNullSpace_ = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(AssembledNullSpaceMap_,); // Nonzeroes abhängig von dim/dofs!!!
       LO iD;
       SC valueTmp;
       GO indices;
       SC values;
       for (UN i=0; i<AssembledNullSpace_->getLocalLength(); i++) {
          iD = rowMap->getLocalElement(AssembledNullSpaceMap_->getGlobalElement(i));
           for (UN j=0; j<AssembledNullSpace_->getNumVectors(); j++) {
               valueTmp=AssembledNullSpace_->getData(j)[i];
               if (fabs(valueTmp)>treshold && iD!=-1) {
                   indices = AssembledNullSpaceMap_->getGlobalElement(j);
                   values= valueTmp;
                   GlobalNullSpace_->replaceGlobalValue(AssembledNullSpaceMap_->getGlobalElement(i),indices,values);
               }
           }
       }*/
       return 0;
   }


    template <class SC,class LO,class GO,class NO>
    int CoarseSpace<SC,LO,GO,NO>::clearCoarseSpace()
    {
        ConstXMapPtrVec emptyVec1;
        UnassembledBasesMaps_.swap(emptyVec1);

        ConstXMapPtrVec emptyVec2;
        UnassembledBasesMapsUnique_.swap(emptyVec2);

        ConstXMultiVectorPtrVec emptyVec3;
        UnassembledSubspaceBases_.swap(emptyVec3);

        AssembledBasisMap_.reset();
        AssembledBasisMapUnique_.reset();
        AssembledBasis_.reset();

        UNVec emptyVec4;
        LocalSubspacesSizes_.swap(emptyVec4);

        GlobalBasisMatrix_.reset();

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int CoarseSpace<SC,LO,GO,NO>::zeroOutBasisVectors(ConstLOVecView zeros)
    {
        FROSCH_ASSERT(!AssembledBasis_.is_null(),"FROSch::CoarseSpace : ERROR: AssembledBasis_.is_null().");
        for (UN j=0; j<AssembledBasis_->getNumVectors(); j++) {
            AssembledBasis_->getVectorNonConst(j)->scale(ScalarTraits<SC>::zero());
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    bool CoarseSpace<SC,LO,GO,NO>::hasUnassembledMaps() const
    {
        return UnassembledBasesMaps_.size()>0;
    }

    template <class SC,class LO,class GO,class NO>
    bool CoarseSpace<SC,LO,GO,NO>::hasBasisMap() const
    {
        return !AssembledBasisMap_.is_null();
    }

    template <class SC,class LO,class GO,class NO>
    typename CoarseSpace<SC,LO,GO,NO>::ConstXMapPtr CoarseSpace<SC,LO,GO,NO>::getBasisMap() const
    {
        FROSCH_ASSERT(!AssembledBasisMap_.is_null(),"FROSch::CoarseSpace : ERROR: AssembledBasisMap_.is_null().");
        return AssembledBasisMap_;
    }

    template <class SC,class LO,class GO,class NO>
    bool CoarseSpace<SC,LO,GO,NO>::hasBasisMapUnique() const
    {
        return !AssembledBasisMapUnique_.is_null();
    }

    template <class SC,class LO,class GO,class NO>
    typename CoarseSpace<SC,LO,GO,NO>::ConstXMapPtr CoarseSpace<SC,LO,GO,NO>::getBasisMapUnique() const
    {
        FROSCH_ASSERT(!AssembledBasisMapUnique_.is_null(),"FROSch::CoarseSpace : ERROR: AssembledBasisMapUnique_.is_null().");
        return AssembledBasisMapUnique_;
    }

    template <class SC,class LO,class GO,class NO>
    bool CoarseSpace<SC,LO,GO,NO>::hasAssembledBasis() const
    {
        return !AssembledBasis_.is_null();
    }

    template <class SC,class LO,class GO,class NO>
    typename CoarseSpace<SC,LO,GO,NO>::ConstXMultiVectorPtr CoarseSpace<SC,LO,GO,NO>::getAssembledBasis() const
    {
        FROSCH_ASSERT(!AssembledBasis_.is_null(),"FROSch::CoarseSpace : ERROR: AssembledBasis_.is_null().");
        return AssembledBasis_;
    }

    template <class SC,class LO,class GO,class NO>
    typename CoarseSpace<SC,LO,GO,NO>::ConstXMultiVectorPtr CoarseSpace<SC,LO,GO,NO>::getAssembledNullSpace() const
    {
        FROSCH_ASSERT(!AssembledBasis_.is_null(),"FROSch::CoarseSpace : ERROR: AssembledBasis_.is_null().");
        return AssembledNullSpace_;
    }


    template <class SC,class LO,class GO,class NO>
    typename CoarseSpace<SC,LO,GO,NO>::ConstUNVecView CoarseSpace<SC,LO,GO,NO>::getLocalSubspaceSizes() const
    {
        return LocalSubspacesSizes_();
    }

    template <class SC,class LO,class GO,class NO>
    bool CoarseSpace<SC,LO,GO,NO>::hasGlobalBasisMatrix() const
    {
        return !GlobalBasisMatrix_.is_null();
    }

    template <class SC,class LO,class GO,class NO>
    typename CoarseSpace<SC,LO,GO,NO>::XMatrixPtr CoarseSpace<SC,LO,GO,NO>::getGlobalBasisMatrix() const
    {
        FROSCH_ASSERT(!GlobalBasisMatrix_.is_null(),"FROSch::CoarseSpace : ERROR: GlobalBasisMatrix_.is_null().");
        return GlobalBasisMatrix_;
    }
}

#endif
