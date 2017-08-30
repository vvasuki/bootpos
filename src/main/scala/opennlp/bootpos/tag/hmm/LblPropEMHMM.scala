package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import opennlp.bootpos.tag.labelPropagation._
import opennlp.bootpos.app._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._
import upenn.junto.app._

class LblPropEMHMM(sentenceSepTagStr :String, sentenceSepWordStr: String, bUseTrainingStats: Boolean = false) extends
EMHMM(sentenceSepTagStr, sentenceSepWordStr, bUseTrainingStats) {
  var lblPropTrainer = new LabelPropagationTrainer(sentenceSepTagStr, sentenceSepWordStr)
  var lblPropTagger: Tagger = lblPropTrainer.tagger
  tagger.intMap = lblPropTagger.intMap
  setIntMap

/*
  Claims:
    Sets up lblPropTagger with dictionary.
    Updates :
        numWordsTraining (Required during evaluation.)
        singletonWordsPerTag [and optionally other counts using dictionary] (required for computing Pr(W|T))
          then scales it down by numWords.
      For completeness, it does other things done by HMM.trainWithDictionary.
        This is because we now have some information to initialize HMM probabilities,
          although we expect untagged data to do label propagation + EM.
        Further, we need to update singletonWordsPerTag.
      See comments for that function too.
  Confidence in correctness: High.
  Reason: Proved correct.
  */
  override def trainWithDictionary(dictionary: Dictionary) = {
    lblPropTagger = lblPropTrainer.trainWithDictionary(dictionary)
    super.trainWithDictionary(dictionary)
    log info "Trained with "+ intMap.numWordsTraining + " words."
    tagger
  }

/*  
  Does label propagation to get a label distribution for the untagged text.
  Uses this to deduce initial parameters to run EM with.
  Confidence in correctness: High.
  Reason: Proved correct.
  */
  override def processUntaggedData(textIn: ArrayBuffer[String]) = {
    val textInUp = textIn.map(_.map(_.toUpper))
    val tokenSeq1 = textInUp.map(x => intMap.getWordId(x))
    val labelDistributions = lblPropTagger.getTagDistributions(tokenSeq1)
    log info "Got label distribution from label propagation."

    // Now forget counts derived from dictionary - but not completely.
    //   (See comments for EMHMM.processTokenSeq)
    wordTagStatsFinal.scaleDown(1/wordTagStatsFinal.numWords.toDouble)

    wordTagStatsFinal.updateCountsPr(tokenSeq1, tagger, labelDistributions)
    log info "Initialized HMM parameters using label distribution."

    // Free memory.
    lblPropTrainer = null
    lblPropTagger = null
    
/*    Ensure that the following, which leads to errors, does not happen above:
    x is tagged sentenceSepTagStr for x != sentenceSepWordStr.*/

    super.processTokenSeq(tokenSeq1)
    tagger
  }

}