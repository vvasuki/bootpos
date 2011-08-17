package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._

class EMHMM(sentenceSepTagStr :String, sentenceSepWordStr: String) extends HMM(sentenceSepTagStr, sentenceSepWordStr){
  var numWordsSeen = 0
  override def processUntaggedData(textIn: ArrayBuffer[String]) = {
    val text = textIn.map(x => getWordId(x))
    numWordsSeen = wordIntMap.size
    val numIterations = 3
    for(i <- 1 to numIterations){
      val wordTagStats = reflectionUtil.deepCopy(wordTagStatsFinal)
      val forwardPr = getForwardPr(text)
      val backwardPr = getBackwardPr(text)
      wordTagStats.updateCounts(forwardPr, backwardPr)
    }
    
  }

//   Assumes that the first and last tokens are equal to sentenceSepWordStr
  def getForwardPr(text: ArrayBuffer[Int]): MatrixBufferDense[Double] = {
    val numTokens = text.length
    val forwardPr = new MatrixBufferDense[Double](numTokens, numTags, -Double.NegativeInfinity)
    forwardPr(0, sentenceSepTag) = 1
    for{i <- 1 to numTokens-1
      token = text(i)
      tag <- wordTagStatsFinal.possibleTags(token)
      prevTag <- wordTagStatsFinal.possibleTags(text(i-1))
    }
    {
      // transition probability given prior tokens
      val transitionPr = forwardPr(token-1, prevTag) + getArcPr(tag, prevTag, token)
      forwardPr(token, tag) = mathUtil.logAdd(forwardPr(token, tag), transitionPr)
    }
    forwardPr
  }

//   Assumes that the first and last tokens are equal to sentenceSepWordStr
  def getBackwardPr(text: ArrayBuffer[Int]): MatrixBufferDense[Double] = {
    val numTokens = text.length
    val backwardPr = new MatrixBufferDense[Double](numTokens, numTags, -Double.NegativeInfinity)
    backwardPr(numTokens-1, sentenceSepTag) = 1
    for{i <- numTokens-1 to 1 by -1
      token = text(i)
      tag <- wordTagStatsFinal.possibleTags(token)
      prevTag <- wordTagStatsFinal.possibleTags(text(i-1))
    }
    {
      // transition probability given succeeding tokens
      val transitionPr = backwardPr(token, prevTag) + getArcPr(tag, prevTag, token)
      backwardPr(token-1, tag) = mathUtil.logAdd(backwardPr(token-1, tag), transitionPr)
    }
    backwardPr
  }
}