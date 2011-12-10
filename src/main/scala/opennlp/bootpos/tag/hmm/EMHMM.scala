package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import opennlp.bootpos.app._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._

class EMHMM(sentenceSepTagStr :String, sentenceSepWordStr: String, bUseTrainingStats: Boolean = false) extends HMMTrainer(sentenceSepTagStr, sentenceSepWordStr){
  override val wordTagStatsFinal = new WordTagStatsProb(intMap.TAGNUM_IN, intMap.WORDNUM_IN, intMap)
  
/*
  Purpose:
    1. Update numWordsSeen.
    1.5 Execute EM algorithm.
    2. Update: logPrTGivenT logPrWGivenT logPrNovelWord
    Ensure EM iterations start with fresh counts when
    starting point has been deduced from a wiktionary.
    
  Confidence: Moderate.
  Reason: See comments for updateCounts. Otherwise proved correct.
*/
  override def processUntaggedData(textIn: ArrayBuffer[String]) = {
    log info "Processing untagged token sequence."
    val text = textIn.map(x => intMap.getWordId(x))
    processTokenSeq(text)
    tagger
  }
  
  def processTokenSeq(text: ArrayBuffer[Int]) = {
    intMap.numWordsSeen = intMap.wordIntMap.size
    val numIterations = BootPos.numIterations
//     Note: wordCount updated using untagged data.
    log info(tagger.toString)
    wordTagStatsFinal.updateWordCount(text, tagger, bUpdateEmissionProb = false)

    log info("\n\nInitial counts:")
    log info(wordTagStatsFinal.toString)
    log info("\n\nInitial params:")
    log info(tagger.toString)
    log info("bUseTrainingStats: " + bUseTrainingStats)
    for(i <- 1 to numIterations){
      var wordTagStats: WordTagStatsProb = null
      wordTagStats = reflectionUtil.deepCopy(wordTagStatsFinal)
      if(!bUseTrainingStats) {
        // wordTagStats = new WordTagStatsProb(intMap.TAGNUM_IN, intMap.WORDNUM_IN)
/*      Doing the above and correctly computing Pr(W|T=t)
          using the tagged-data learning code can be tricky.
        Also it may not be desirable to totally forget things learned
          with either the trainingStats (which possibly is from a dictionary.)
        
        Note that singletonWordsPerTag is not touched below -
          If needed it should be scaled down elsewhere.
        
        Hence, we do the below.
        The fraction 1/wordTagStats.numWords is arbitrarily chosen.
        Could perhaps have used wordTagStats.tagCount.sum.toDouble.
        Smaller the fraction, lesser the weight given to training counts.
*/
        wordTagStats.scaleDown(1/wordTagStats.tagCount.sum.toDouble)
        log info("Not using training stats.")
        log info(wordTagStats.toString)
      }
      log info("Iteration: " + i)
      val forwardPr = getForwardPr(text)
      val backwardPr = getBackwardPr(text)
      wordTagStats.updateCountsEM(text, tagger, forwardPr, backwardPr)
/*      log info("\n\nParams:")
      log info(tagger)*/
    }
  }
  
/*
  Assumes that the first and last tokens are equal to sentenceSepWordStr
  @return forwardPr.
  Confidence: High
  Reason: Proved correct.
    Also, output verified on a test case.
  */
  def getForwardPr(text: ArrayBuffer[Int]): MatrixBufferDense[Double] = {
    val numTokens = text.length
    val forwardPr = new MatrixBufferDense[Double](numTokens, numTags, math.log(0), bSetInitSize = true)
    forwardPr(0, intMap.sentenceSepTag) = math.log(1)
    for{i <- 1 to numTokens-1
      token = text(i)
      tag <- intMap.possibleTags(token)
      prevTag <- intMap.possibleTags(text(i-1))
    }
    {
      // transition probability given prior tokens
      val transitionPr = forwardPr(i-1, prevTag) + tagger.getArcPr(tag, prevTag, token)
      forwardPr(i, tag) = mathUtil.logAdd(forwardPr(i, tag), transitionPr)
//       if(i> 2725 && i<2835)
      if(forwardPr(i, tag) == math.log(0))
//       if(false)
      {
        log info("i "+i + " token "+token + " prevTag "+ prevTag + " tag " + tag)
        log info("wordTagCount(token, tag) " + wordTagStatsFinal.wordTagCount(token, tag) +
        " logPrWGivenT(token, tag) " + tagger.logPrWGivenT(token, tag)+
        " trPr "+ transitionPr)
        log info("forwardPr(i-1, prevTag) "+ forwardPr(i-1, prevTag))
        log info("arcPr "+ tagger.getArcPr(tag, prevTag, token))
        log info("forwardPr(i, tag) " + forwardPr(i, tag))
      }
    }
//     log info(forwardPr.matrix.zipWithIndex)
    forwardPr
  }

/*
  Assumes that the first and last tokens are equal to sentenceSepWordStr
  @return backwardPr.
  Confidence: High
  Reason: Proved correct.
    Also output verified on a test case.
  */
  def getBackwardPr(text: ArrayBuffer[Int]): MatrixBufferDense[Double] = {
    val numTokens = text.length
    val backwardPr = new MatrixBufferDense[Double](numTokens, numTags, math.log(0), bSetInitSize = true)
    backwardPr(numTokens-1, intMap.sentenceSepTag) = math.log(1)
    for{i <- numTokens-1 to 1 by -1
      token = text(i)
      tag <- intMap.possibleTags(token)
      prevTag <- intMap.possibleTags(text(i-1))
    }
    {
      // transition probability given succeeding tokens
      val transitionPr = backwardPr(i, tag) + tagger.getArcPr(tag, prevTag, token)
      backwardPr(i-1, prevTag) = mathUtil.logAdd(backwardPr(i-1, prevTag), transitionPr)
    }
    // log info(backwardPr.map(math.exp(_)))
    backwardPr
  }
}