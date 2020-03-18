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

#ifndef _FROSCH_PARTITIONOFUNITYBASIS_DEF_hpp
#define _FROSCH_PARTITIONOFUNITYBASIS_DEF_hpp

#include <FROSch_LocalPartitionOfUnityBasis_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template<class SC,class LO,class GO,class NO>
    LocalPartitionOfUnityBasis<SC,LO,GO,NO>::LocalPartitionOfUnityBasis(CommPtr mpiComm,
                                                                        CommPtr serialComm,
                                                                        UN dofsPerNode,
                                                                        ParameterListPtr parameterList,
                                                                        ConstXMultiVectorPtr nullSpaceBasis,
                                                                        XMultiVectorPtrVecPtr partitionOfUnity,
                                                                        XMapPtrVecPtr partitionOfUnityMaps) :
    MpiComm_ (mpiComm),
    SerialComm_ (serialComm),
    DofsPerNode_ (dofsPerNode),
    ParameterList_ (parameterList),
    PartitionOfUnity_ (partitionOfUnity),
    NullspaceBasis_ (nullSpaceBasis),
    PartitionOfUnityMaps_ (partitionOfUnityMaps)
    {

    }

    template<class SC,class LO,class GO,class NO>
    int LocalPartitionOfUnityBasis<SC,LO,GO,NO>::addPartitionOfUnity(XMultiVectorPtrVecPtr partitionOfUnity,
                                                                     XMapPtrVecPtr partitionOfUnityMaps)
    {
        PartitionOfUnity_ = partitionOfUnity;
        PartitionOfUnityMaps_ = partitionOfUnityMaps;
        return 0;
    }

    template<class SC,class LO,class GO,class NO>
    int LocalPartitionOfUnityBasis<SC,LO,GO,NO>::addGlobalBasis(ConstXMultiVectorPtr nullSpaceBasis)
    {
        NullspaceBasis_ = nullSpaceBasis;
        return 0;
    }


    template<class SC,class LO,class GO,class NO>
    int LocalPartitionOfUnityBasis<SC,LO,GO,NO>::buildLocalPartitionOfUnityBasis()
    {
        FROSCH_ASSERT(!NullspaceBasis_.is_null(),"Nullspace Basis is not set.");
        FROSCH_ASSERT(!PartitionOfUnity_.is_null(),"Partition Of Unity is not set.");
        FROSCH_ASSERT(!PartitionOfUnityMaps_.is_null(),"Partition Of Unity Map is not set.");

        LocalPartitionOfUnitySpace_ = CoarseSpacePtr(new CoarseSpace<SC,LO,GO,NO>(this->MpiComm_,this->SerialComm_));
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

        Teuchos::Array<Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<LO,SC> > > > tmpCBasis(PartitionOfUnity_.size());
        Teuchos::Array<Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<LO,SC> > > > tmpCBasisR(PartitionOfUnity_.size());

        XMultiVectorPtrVecPtr2D tmpBasis(PartitionOfUnity_.size());
        XMultiVectorPtrVecPtr2D tmpBasisR(PartitionOfUnity_.size());
        ConstXMapPtr nullspaceBasisMap = NullspaceBasis_->getMap();
        for (UN i=0; i<PartitionOfUnity_.size(); i++) {
            if (!PartitionOfUnity_[i].is_null()) {
                FROSCH_ASSERT(PartitionOfUnityMaps_[i]->getNodeNumElements()>0,"PartitionOfUnityMaps_[i]->getNodeNumElements()==0");
                tmpBasis[i] = XMultiVectorPtrVecPtr(PartitionOfUnity_[i]->getNumVectors());
                tmpBasisR[i] = XMultiVectorPtrVecPtr(PartitionOfUnity_[i]->getNumVectors());

                tmpCBasis[i] = Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<LO,SC> > > (PartitionOfUnity_[i]->getNumVectors());
                tmpCBasisR[i] = Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<LO,SC> > > (PartitionOfUnity_[i]->getNumVectors());

                for (UN j=0; j<PartitionOfUnity_[i]->getNumVectors(); j++) {
                    XMultiVectorPtr tmpBasisJ = MultiVectorFactory<SC,LO,GO,NO>::Build(NullspaceBasis_->getMap(),NullspaceBasis_->getNumVectors());
                    XMapPtr CMap = MapFactory<LO,GO,NO>::Build(NullspaceBasis_->getMap()->lib(),NullspaceBasis_->getNumVectors(),0,SerialComm_);
                    XMultiVectorPtr CbasisR = MultiVectorFactory<SC,LO,GO,NO>::Build(CMap,NullspaceBasis_->getNumVectors());
                    tmpBasisJ->elementWiseMultiply(ScalarTraits<SC>::one(),*PartitionOfUnity_[i]->getVector(j),*NullspaceBasis_,ScalarTraits<SC>::one());
                    Teuchos::SerialDenseMatrix<LO,SC> tmpCBasisJ(NullspaceBasis_->getMap()->getNodeNumElements(),NullspaceBasis_->getNumVectors());
                    XMultiVectorPtr tmpR;
                    if (ParameterList_->get("Orthogonalize",true)) {
                        tmpBasis[i][j] = ModifiedGramSchmidt(tmpBasisJ.getConst());

                    } else {
                        tmpBasis[i][j] = tmpBasisJ;
                    }
                    tmpR = ModGram_FormR(tmpBasisJ.getConst(),tmpBasis[i][j].getConst());
                    //if(MpiComm_->getRank() == 0)tmpR->describe(*fancy,Teuchos::VERB_EXTREME);
                //tmpBasis[i][j] = MultiVectorFactory<SC,LO,GO,NO>::Build(NullspaceBasis_->getMap(),NullspaceBasis_->getNumVectors());

                     for(UN h = 0;h<NullspaceBasis_->getNumVectors();h++){
                          Teuchos::ArrayRCP<SC> data = tmpBasisJ->getDataNonConst(h);
                          for(UN k = 0;k<NullspaceBasis_->getMap()->getNodeNumElements();k++){
                            tmpCBasisJ(k,h) = data[k];
                        }
                    }
                    Teuchos::SerialQRDenseSolver<LO,SC> qrSolver;
                    qrSolver.setMatrix(Teuchos::rcp(&tmpCBasisJ, false));
                     if (ParameterList_->get("Orthogonalize Coarse",true)) {
                           qrSolver.factor();
                           qrSolver.formQ();
                           qrSolver.formR();
                           tmpCBasis[i][j]  = qrSolver.getQ();
                           tmpCBasisR[i][j] = qrSolver.getR();
                          /* if(MpiComm_->getRank() == 0){
                              tmpCBasis[i][j]->print(std::cout);
                              tmpCBasisR[i][j]->print(std::cout);
                           }*/
                    }
                    Teuchos::SerialDenseMatrix<LO,SC> transMat (*tmpCBasis[i][j],Teuchos::NO_TRANS);
                    Teuchos::SerialDenseMatrix<LO,SC> transRMat (*tmpCBasisR[i][j],Teuchos::NO_TRANS);
                    UN zero_count = 0;
                    Teuchos::SerialDenseVector<LO,SC>  densvec (NullspaceBasis_->getNumVectors());
                    Teuchos::Array<UN> nullcol(CMap->getNodeNumElements());
                    for(UN z = 0;z<CMap->getNodeNumElements();z++){
                      for(UN h = 0;h<NullspaceBasis_->getNumVectors();h++){
                        CbasisR->replaceLocalValue(z,h,transRMat(z,h));
                        densvec(h) = transRMat(z,h);
                        //if(transRMat(z,h)<= 1.0e-10) zero_count++;
                      }
                      if(densvec.normFrobenius()< 1.0e-10)nullcol[z] = 1;
                      //if(zero_count == NullspaceBasis_->getNumVectors()) nullcol[z] = 1;
                      else nullcol[z] = 0;

                      zero_count =0;
                    }

                    for(UN h = 0;h<NullspaceBasis_->getNumVectors();h++){
                      for(UN z = 0;z<NullspaceBasis_->getMap()->getNodeNumElements();z++){
                          if(nullcol[h] == 0){
                            //tmpBasis[i][j]->replaceLocalValue(z,h,transMat(z,h));
                          }//else{
                            //tmpBasis[i][j]->replaceLocalValue(z,h,ScalarTraits<SC>::zero());
                          //}
                      }
                    }
                   tmpBasisR[i][j]=tmpR;
                }

            } else {
                FROSCH_ASSERT(PartitionOfUnityMaps_[i]->getNodeNumElements()==0,"PartitionOfUnityMaps_[i]->getNodeNumElements()!=0");
            }
        }

        UNVecPtr maxNV(PartitionOfUnity_.size());
      for (UN i=0; i<PartitionOfUnity_.size(); i++) {
          UN maxNVLocal = 0;
          if (!PartitionOfUnityMaps_[i].is_null()) {
              if (!PartitionOfUnity_[i].is_null()) {
                  for (UN j=0; j<tmpBasis[i].size(); j++) {
                      maxNVLocal = std::max(maxNVLocal,(UN) tmpBasis[i][j]->getNumVectors());
                  }
              }
              reduceAll(*MpiComm_,REDUCE_MAX,maxNVLocal,ptr(&maxNV[i]));
          }
      }

      if (!ParameterList_->get("Number of Basisfunctions per Entity","MaxAll").compare("MaxAll")) {
          UNVecPtr::iterator max = std::max_element(maxNV.begin(),maxNV.end());
          for (UN i=0; i<maxNV.size(); i++) {
              maxNV[i] = *max;
          }
      } else if (!ParameterList_->get("Number of Basisfunctions per Entity","MaxAll").compare("MaxEntityType")) {

      } else {
          FROSCH_ASSERT(false,"Number of Basisfunctions per Entity type is unknown.");
      } // Testen!!!!!!!!!!!!!!!!!!!!!!!! AUSGABE IMPLEMENTIEREN!!!!!!

        // Kann man das schöner machen?
        for (UN i=0; i<PartitionOfUnity_.size(); i++) {
            if (!PartitionOfUnityMaps_[i].is_null()) {
                if (!PartitionOfUnity_[i].is_null()) {
                    for (UN j=0; j<maxNV[i]; j++) {
                        //XMultiVectorPtrVecPtr tmpBasis2(PartitionOfUnity_[i]->getNumVectors());

                        XMultiVectorPtr entityBasis = MultiVectorFactory<SC,LO,GO,NO >::Build(PartitionOfUnity_[i]->getMap(),PartitionOfUnity_[i]->getNumVectors());
                        XMapPtr CNullMap = MapFactory<LO,GO,NO>::Build(NullspaceBasis_->getMap()->lib(),NullspaceBasis_->getNumVectors(),0,SerialComm_);

                        XMultiVectorPtr entityBasisR = MultiVectorFactory<SC,LO,GO,NO >::Build(CNullMap,PartitionOfUnity_[i]->getNumVectors());
                        entityBasis->scale(ScalarTraits<SC>::zero());
                        for (UN k=0; k<PartitionOfUnity_[i]->getNumVectors(); k++) {
                            if (j<tmpBasis[i][k]->getNumVectors()) {
                                entityBasis->getDataNonConst(k).deepCopy(tmpBasis[i][k]->getData(j)()); // Here, we copy data. Do we need to do this?
                            }
                            if (j<tmpBasisR[i][k]->getNumVectors()) {
                                entityBasisR->getDataNonConst(k).deepCopy(tmpBasisR[i][k]->getData(j)());
                            }
                        }
                        LocalPartitionOfUnitySpace_->addSubspace(PartitionOfUnityMaps_[i],null,entityBasis);
                        if (ParameterList_->get("Coarse NullSpace",false)) {
                           LocalPartitionOfUnitySpace_->addNullspace(PartitionOfUnityMaps_[i],entityBasisR);
                        }
                    }
                } else {
                    for (UN j=0; j<NullspaceBasis_->getNumVectors(); j++) {
                        LocalPartitionOfUnitySpace_->addSubspace(PartitionOfUnityMaps_[i]);
                    }
                }
            } else {
                FROSCH_WARNING("FROSch::LocalPartitionOfUnityBasis",this->MpiComm_->getRank()==0,"PartitionOfUnityMaps_[i].is_null()");
            }
        }
        UN maxNumBasis = *std::max_element(maxNV.begin(),maxNV.end());
        LocalPartitionOfUnitySpace_->assembleCoarseSpace();
        if (ParameterList_->get("Coarse NullSpace",false)) {
           LocalPartitionOfUnitySpace_->assembleNullSpace(maxNumBasis);
           CoarseNullSpace_ = LocalPartitionOfUnitySpace_->getAssembledNullSpace();
         }

        return 0;
    }

    template<class SC,class LO,class GO,class NO>
    typename LocalPartitionOfUnityBasis<SC,LO,GO,NO>::XMultiVectorPtrVecPtr LocalPartitionOfUnityBasis<SC,LO,GO,NO>::getPartitionOfUnity() const
    {
        FROSCH_ASSERT(!PartitionOfUnity_.is_null(),"Partition Of Unity is not set.");
        return PartitionOfUnity_;
    }

    template<class SC,class LO,class GO,class NO>
    typename LocalPartitionOfUnityBasis<SC,LO,GO,NO>::XMultiVectorPtr LocalPartitionOfUnityBasis<SC,LO,GO,NO>::getNullspaceBasis() const
    {
        FROSCH_ASSERT(!NullspaceBasis_.is_null(),"Nullspace Basis is not set.");
        return NullspaceBasis_;
    }

    template<class SC,class LO,class GO,class NO>
    typename LocalPartitionOfUnityBasis<SC,LO,GO,NO>::ConstXMultiVectorPtr LocalPartitionOfUnityBasis<SC,LO,GO,NO>::getCoarseNullSpace() const
    {
        FROSCH_ASSERT(!CoarseNullSpace_.is_null(),"Nullspace Basis is not set.");
        return CoarseNullSpace_;
    }

    template<class SC,class LO,class GO,class NO>
    typename LocalPartitionOfUnityBasis<SC,LO,GO,NO>::CoarseSpacePtr LocalPartitionOfUnityBasis<SC,LO,GO,NO>::getLocalPartitionOfUnitySpace() const
    {
        FROSCH_ASSERT(!LocalPartitionOfUnitySpace_.is_null(),"Local Partition Of Unity Space is not built yet.");
        return LocalPartitionOfUnitySpace_;
    }
}

#endif
