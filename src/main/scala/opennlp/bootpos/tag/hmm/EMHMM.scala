package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import opennlp.bootpos.app._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._

class WordTagStatsProb(TAGNUM_IN: Int, WORDNUM_IN: Int) extends WordTagStats(TAGNUM_IN, WORDNUM_IN){
/*
  Example against which much of this code was verified:
    http://comp.ling.utexas.edu/_media/courses/2008/fall/natural_language_processing/eisner-icecream-forwardbackward.xls
  Purpose:
      0. Execute forward/ backward algorithm.
      1. Update wordTagCount, tagBeforeTagCount, tagCount, tokenCount
      2. Update: logPrTagGivenTag logPrWordGivenTag logPrNovelWord
  Confidence: High.
  Reason: Proved correct. Also verified with ic test data.
    See comments below.
*/
  def updateCounts(text: ArrayBuffer[Int], hmm: EMHMM) = {
    val numTokens = text.length
    val sentenceSepTag = hmm.sentenceSepTag

    val forwardPr = hmm.getForwardPr(text)
    val backwardPr = hmm.getBackwardPr(text)
    val numTokensUntagged = forwardPr.length
    val wordTagStatsFinal = hmm.wordTagStatsFinal

    val prTokens = forwardPr(numTokensUntagged-1, sentenceSepTag)

    prepareTableSizes(hmm.numWordsTotal, wordTagStatsFinal.numTags)
/*
    Claim: wordTagCount, tagCount correctly updated below.
    Confidence: Moderate.
    Reason: Not sure whether underflow errors occur.
      Otherwise proved correct.
*/
    for{i <- 1 to numTokens-1
      token = text(i)
      tag <- wordTagStatsFinal.possibleTags(token, hmm)
    }{
      val prTag = forwardPr(i, tag) + backwardPr(i, tag) - prTokens
      wordTagCount(token, tag) = wordTagCount(token, tag) + math.exp(prTag)
      tagCount(tag) = tagCount(tag) + math.exp(prTag)
    }

/*
    Claim: tagBeforeTagCount correctly updated below.
    Confidence: Moderate.
    Reason: Not sure whether underflow errors occur.
      Otherwise proved correct.*/
    for{i <- 1 to numTokens-1
      token = text(i)
      tag <- wordTagStatsFinal.possibleTags(token, hmm)
      prevTag <- wordTagStatsFinal.possibleTags(text(i-1), hmm)
    }{
      val prTagPair = forwardPr(i-1, prevTag) + backwardPr(i, tag) - prTokens + hmm.getArcPr(tag, prevTag, token)
      tagBeforeTagCount(prevTag, tag) = tagBeforeTagCount(prevTag, tag) + math.exp(prTagPair)
    }

    // println(this)

    setLogPrTagGivenTag(hmm)
    setLogPrWordGivenTag(hmm)
  }

}

class EMHMM(sentenceSepTagStr :String, sentenceSepWordStr: String) extends HMM(sentenceSepTagStr, sentenceSepWordStr){
  var numWordsSeen = 0
  override val wordTagStatsFinal = new WordTagStatsProb(TAGNUM_IN, WORDNUM_IN)
  
/*
  Purpose:
    1. Update numWordsSeen.
    1.5 Execute EM algorithm.
    2. Update: logPrTagGivenTag logPrWordGivenTag logPrNovelWord
  Confidence: Moderate.
  Reason: See comments for updateCounts. Otherwise proved correct.
*/
  override def processUntaggedData(textIn: ArrayBuffer[String]) = {
    val text = textIn.map(x => getWordId(x))
    numWordsSeen = wordIntMap.size
    val numIterations = BootPos.numIterations
    val bUseTrainingStats = true
//     Note: wordCount updated using untagged data.
    wordTagStatsFinal.updateWordCount(text, this)

/*    println("\n\nInitial counts:")
    println(wordTagStatsFinal)
    println("\n\nInitial params:")
    println(this)*/
    for(i <- 1 to numIterations){
      var wordTagStats: WordTagStatsProb = null
      if(bUseTrainingStats)
        wordTagStats = reflectionUtil.deepCopy(wordTagStatsFinal)
      else
        wordTagStats = new WordTagStatsProb(TAGNUM_IN, WORDNUM_IN)
      println("Iteration: " + i)
      wordTagStats.updateCounts(text, this)
/*      println("\n\nParams:")
      println(this)*/
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
    forwardPr(0, sentenceSepTag) = math.log(1)
    for{i <- 1 to numTokens-1
      token = text(i)
      tag <- wordTagStatsFinal.possibleTags(token, this)
      prevTag <- wordTagStatsFinal.possibleTags(text(i-1), this)
    }
    {
      // transition probability given prior tokens
      // println("i "+i + " token "+token + " prevTag "+ prevTag)
      val transitionPr = forwardPr(i-1, prevTag) + getArcPr(tag, prevTag, token)
      forwardPr(i, tag) = mathUtil.logAdd(forwardPr(i, tag), transitionPr)
    }
//     println(forwardPr.map(math.exp(_)))
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
    backwardPr(numTokens-1, sentenceSepTag) = math.log(1)
    for{i <- numTokens-1 to 1 by -1
      token = text(i)
      tag <- wordTagStatsFinal.possibleTags(token, this)
      prevTag <- wordTagStatsFinal.possibleTags(text(i-1), this)
    }
    {
      // transition probability given succeeding tokens
      val transitionPr = backwardPr(i, tag) + getArcPr(tag, prevTag, token)
      backwardPr(i-1, prevTag) = mathUtil.logAdd(backwardPr(i-1, prevTag), transitionPr)
    }
    // println(backwardPr.map(math.exp(_)))
    backwardPr
  }
}