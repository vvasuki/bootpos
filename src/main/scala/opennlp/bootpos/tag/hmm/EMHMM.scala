package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import opennlp.bootpos.app._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._

class EMHMM(sentenceSepTagStr :String, sentenceSepWordStr: String, bUseTrainingStats: Boolean = true) extends HMM(sentenceSepTagStr, sentenceSepWordStr){
  var numWordsSeen = 0
  override val wordTagStatsFinal = new WordTagStatsProb(TAGNUM_IN, WORDNUM_IN)
  
/*
  Purpose:
    1. Update numWordsSeen.
    1.5 Execute EM algorithm.
    2. Update: logPrTagGivenTag logPrWordGivenTag logPrNovelWord
    Ensure EM iterations start with fresh counts when
    starting point has been deduced from a wiktionary.
    
  Confidence: Moderate.
  Reason: See comments for updateCounts. Otherwise proved correct.
*/
  override def processUntaggedData(textIn: ArrayBuffer[String]) = {
    val text = textIn.map(x => getWordId(x))
    numWordsSeen = wordIntMap.size
    val numIterations = BootPos.numIterations
//     Note: wordCount updated using untagged data.
    wordTagStatsFinal.updateWordCount(text, this)

/*    println("\n\nInitial counts:")
    println(wordTagStatsFinal)
    println("\n\nInitial params:")
    println(this)*/
    println("bUseTrainingStats: " + bUseTrainingStats)
    for(i <- 1 to numIterations){
      var wordTagStats: WordTagStatsProb = null
      wordTagStats = reflectionUtil.deepCopy(wordTagStatsFinal)
      if(!bUseTrainingStats) {
        // wordTagStats = new WordTagStatsProb(TAGNUM_IN, WORDNUM_IN)
/*        Doing the above and correctly computing Pr(W|T=t)
        using the tagged-data learning code can be tricky.
        Also it may not be desirable to totally forget things learned
        with either the trainingStats (which possibly is from a dictionary.)*/
        wordTagStats.scaleDown(1/wordTagStats.numWords.toDouble)
        println(wordTagStats)
      }
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