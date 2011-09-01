package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import scala.collection.mutable.HashMap
import scala.collection.immutable.Set
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._

class HMM(sentenceSepTagStr :String, sentenceSepWordStr: String) extends Tagger{
//   Probabilities are stored in log space to avoid underflow.
// logPrTagGivenTag, due to its small size, should be precomputed.
  var logPrTagGivenTag = new MatrixBufferDense[Double](TAGNUM_IN, TAGNUM_IN)
/*
  Considerations in deciding whether to compute logPrWordGivenTag:
  1. Test set may not contain many words seen in training set.
  2. We may want to avoid repeated computation for words which appear multiple times.
  My guess is that [2] outweighs [1].
*/
  var logPrWordGivenTag = new MatrixBufferDense[Double](WORDNUM_IN, TAGNUM_IN, defaultValue = math.log(0))
  var logPrNovelWord = new ExpandingArray[Double](TAGNUM_IN, defaultValue = math.log(0))
  var numWordsTraining = 0

  val wordTagStatsFinal = new WordTagStats(TAGNUM_IN, WORDNUM_IN)
  val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)



//   Confidence: High.
//   Reason: Proved correct.
  def getPrWordGivenTag(word: Int, tag: Int) = {
    if(word < logPrWordGivenTag.numRows) logPrWordGivenTag(word, tag)
    else logPrNovelWord(tag)
  }

//   Confidence: High.
//   Reason: Proved correct.
  def getArcPr(tag:Int, prevTag: Int, word: Int) = {
    logPrTagGivenTag(tag, prevTag) + getPrWordGivenTag(word, tag)
  }

//   Confidence: High.
//   Reason: Proved correct.
  def checkLogPrWGivenT(tag: Int) = {
    mathUtil.logAdd(logPrWordGivenTag.colFold(math.log(0))(tag, mathUtil.logAdd), logPrNovelWord(tag))
  }

//   Confidence: High.
//   Reason: Proved correct.
  override def toString = {
    val randWord = (math.random * logPrWordGivenTag.numRows).toInt
    val randTag = (math.random * logPrTagGivenTag.numRows).toInt
    var str = "t="+randTag + " w="+randWord
/*    str +=("\nT|T " + logPrTagGivenTag.map(math.exp))
    str +=("\nW=w|T " + logPrWordGivenTag(randWord).map(math.exp))*/
/*    str += "\n sum T|T=t " + (logPrTagGivenTag.colFold(math.log(0))(randTag, mathUtil.logAdd))
    str += "\n sum W|T=t " + checkLogPrWGivenT(randTag)*/
    str +=("\nNW|T " + logPrNovelWord)
    str +=("\nsum_W Pr(W|T)" + (0 to numTags-1).map(checkLogPrWGivenT(_)))
    str
  }



/*
Claims.
Correctly updates the following:
  wordIntMap, tagIntMap.
  numWordsTraining
  tagCount, wordTagCount, singletonWordsPerTag.
  logPrTagGivenTag
  logPrNovelWord
  logPrWordGivenTag
*/
//  Confidence in correctness: High.
//  Reason: Well tested.
  def train(iter: Iterator[Array[String]]) = {
    val lstData = iter.map(x => Array(getWordId(x(0)), getTagId(x(1)))).toList
    wordTagStatsFinal.updateCounts(lstData, this)
    numWordsTraining = numWordsTotal
/*    println(wordTagStatsFinal)
    println(this)*/
  }

/*
Claims.
Correctly updates the following:
  wordIntMap, tagIntMap.
  numWordsTraining
  tagCount, wordTagCount, singletonWordsPerTag.
  logPrTagGivenTag
  logPrNovelWord
  logPrWordGivenTag
  
Problems to consider when creating an initial HMM model from a dictionary.
1] The dictionary may be incomplete in two ways:
  1a] There may be missing words.
  1b] Even when a word is included, some potential parts of speech may not be mentioned.
2] We have no word-sequence and tag-sequence information.

Potential model:
Let W be the set of words in the dictionary.
Let W_t be the set of words with potential tag t.
Pr(t_i|t_{i-1}) = Pr(t_i) - or perhaps just 1/numTags.
  The latter choice may be preferable because it may
  facilitate better learning of Pr(t_i|t_{i-1}) during EM.
Model Pr(w \notin W) arbitrarily.
Set Pr(w \notin W_t) = Pr(w \notin W)
Pr(w \in W|t) = Pr(w | w \in W_t) Pr(w \in W_t|t).

Tag count should be updated during HMM.
Ensure EM iterations start with fresh counts when starting point has been deduced from a wiktionary.
*/
//  Confidence in correctness: Medium.
//  Reason: Seems to be fine.
  override def trainWithDictionary(dictionary: Dictionary) = {
    var lstData = dictionary.lstData.
    map(x => Array(getWordId(x(0)), getTagId(x(1))))
    numWordsTraining = numWordsTotal
    
    wordTagStatsFinal.updateWordTagCount(lstData.toList)

    logPrTagGivenTag = new MatrixBufferDense[Double](numTags, numTags, math.log(1/numTags.toDouble), true)
    wordTagStatsFinal.setLogPrWordGivenTag(this, dictionary)
    println("tokens in training data: " + lstData.length)
    println(wordTagStatsFinal)
    println(this)
  }
  
//  Confidence in correctness: High.
//  Reason: Well tested.
  def test(testDataIn: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
    println(wordTagStatsFinal)
    println(this)
    val testData = testDataIn.map(x => Array(getWordId(x(0)), getTagId(x(1))))
    val numTokens = testData.length
    val numTags = wordTagStatsFinal.numTags;
    var resultPair = new ArrayBuffer[Array[Boolean]](numTokens)
    resultPair = resultPair.padTo(numTokens, null)

    var bestPrevTag = new MatrixBufferDense[Int](numTokens + 1, numTags)
    var logPrSequence = new MatrixBufferDense[Double](numTokens + 1, numTags, defaultValue=math.log(0))
    var bSeekSentence = true
    logPrSequence(0, sentenceSepTag) = math.log(1)
    for{tokenNum <- 1 to numTokens;
        token = testData(tokenNum-1)(0)
        tag <- wordTagStatsFinal.possibleTags(token, this)
    }{
      val logPrW = getPrWordGivenTag(token, tag)
      var logPrJ = matrixMath.vp(logPrSequence(tokenNum-1), logPrW)
//      Ensure that perplexity is not affected by empty sentences.
      if(!(bSeekSentence && token==sentenceSepWord))
        logPrJ = matrixMath.vp(logPrJ, logPrTagGivenTag(tag))
      logPrSequence(tokenNum, tag) = logPrJ.max
      bestPrevTag(tokenNum, tag) = logPrJ.indexOf(logPrSequence(tokenNum, tag))

      bSeekSentence = token == sentenceSepWord

//      println("logPrSeq "+ logPrSequence(tokenNum))
//      println("# "+tokenNum + " w " + token + " tg "+ tag + " tg_{-1} "+ bestPrevTag(tokenNum, tag))
    }

    val bestTags = new Array[Int](numTokens)
    bestTags(numTokens-1) = logPrSequence(numTokens).indexOf(logPrSequence(numTokens).max)
    resultPair(numTokens-1) = Array(testData(numTokens-1)(1) == bestTags(numTokens-1), testData(numTokens-1)(0) >= numWordsTraining)
    var perplexity = math.exp(-logPrSequence(numTokens, bestTags(numTokens-1))/numTokens)
    println("Perplexity: " + perplexity)

    for(tokenNum <- numTokens-2 to 0 by -1) {
      var token = testData(tokenNum)(0)
      bestTags(tokenNum) = bestPrevTag(tokenNum+2, bestTags(tokenNum+1))
      val bNovel = token >= numWordsTraining
      resultPair(tokenNum) = Array(bestTags(tokenNum) == testData(tokenNum)(1), bNovel)
      
//      println(tokenNum + " : " + token + " : "+ resultPair(tokenNum))
    }
    resultPair
  }

}
