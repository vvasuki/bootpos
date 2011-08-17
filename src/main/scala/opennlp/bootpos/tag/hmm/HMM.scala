package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._

class HMM(sentenceSepTagStr :String, sentenceSepWordStr: String) extends Tagger{
  val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)

/*
Considerations while precomputing logPrWordGivenTag:
  1. Test set may not contain many words seen in training set.
  2. We may want to avoid repeated computation for words which appear multiple times.
logPrTagGivenTag, due to its small size, should be precomputed.
  */
//   Probabilities are stored in log space to avoid underflow.
  var logPrTagGivenTag = new MatrixBufferDense[Double](TAGNUM_IN, TAGNUM_IN)
  var logPrWordGivenTag = new MatrixBufferDense[Double](WORDNUM_IN, TAGNUM_IN)
  var logPrNovelWord = new ExpandingArray[Double](TAGNUM_IN)
  var numWordsTraining = 0

  var wordTagStatsFinal = new WordTagStats

  def getArcPr(tag:Int, prevTag: Int, word: Int) = {
    var logPrW = logPrNovelWord(tag)
    if(word < numWordsTraining) logPrW = logPrWordGivenTag(word, tag)
    logPrTagGivenTag(tag, prevTag) + logPrW
  }

  class WordTagStats {
  // The following are Double arrays because in case of EM-HMM, counts could be a non-integer.
    var wordTagCount = new MatrixBufferDense[Double](WORDNUM_IN, TAGNUM_IN)
    var singletonWordsPerTag = new ExpandingArray[Double](TAGNUM_IN)
    var tagBeforeTagCount = new MatrixBufferDense[Double](TAGNUM_IN, TAGNUM_IN)
    var tagCount = new ExpandingArray[Double](TAGNUM_IN)
    var tokenCount = 0.0
    
    def numWords = wordTagCount.length
    def numTags = tagBeforeTagCount.length;

    /*
    Claims.
    Correctly updates the following:
    wordIntMap, tagIntMap.
    numWordsTraining
    tagCount, wordTagCount.
    tokenCount.
    */
    //  Confidence in correctness: High.
    //  Reason: Well tested.
    def updateCounts(iter: Iterator[Array[String]]) = {
      var prevTag = sentenceSepTag
      for(fields <- iter.map(x => Array(getWordId(x(0)), getTagId(x(1))))){
        var tag = fields(1); var word = fields(0)
  //      println(prevTag+ " t " + tag + " w "+ word)
        wordTagCount.increment(word, tag)
        wordTagCount(word, tag) match {
          case 1 => {singletonWordsPerTag.addAt(tag, 1)}
          case 2 => {singletonWordsPerTag.addAt(tag, -1)}
          case _ => {}
        }

        tagBeforeTagCount.increment(prevTag, tag)
        tagCount.addAt(tag, 1)
        prevTag = tag
      }
      tagBeforeTagCount(sentenceSepTag, sentenceSepTag) = 0
      tokenCount = tagCount.sum
      setLogPrTagGivenTag
      setLogPrWordGivenTag
      setLogPrNovelWord
      numWordsTraining = wordTagCount.length
    }

    def updateCounts(forwardPr: MatrixBufferDense[Double], backwardPr: MatrixBufferDense[Double]) = {
      setLogPrTagGivenTag
      setLogPrWordGivenTag
      setLogPrNovelWord
      throw new Exception("implementation incomplete")
    }

    def setLogPrNovelWord = {
      logPrNovelWord.padTill(numTags, math.log(0))
      for(tag<- (0 to numTags-1)) {
        var s = singletonWordsPerTag(tag)+ 1e-100
        var x = (s/(tokenCount + numWords + 1).toDouble)/(s + tagCount(tag).toDouble)
        logPrNovelWord(tag) = math.log(x)
      }
    }

    //  Confidence in correctness: High.
    //  Reason: Well tested.
    def setLogPrTagGivenTag = {
      for(tag1 <- (0 to numTags-1); tag2 <- (0 to numTags-1)) {
        var s = tagBeforeTagCount(tag2).count(x => x==1) + 1e-100
        var x = (tagBeforeTagCount(tag2, tag1) + s*tagCount(tag1)/tokenCount.toDouble)/(tagBeforeTagCount(tag2).sum + s).toDouble
        logPrTagGivenTag(tag1, tag2) = math.log(x)
  //       println(tag1 + "|" + tag2+ " = " + x)
      }
    }

  //  Confidence in correctness: High.
  //  Reason: Well tested.
    def setLogPrWordGivenTag = {
      for(tag <- (0 to numTags-1); word <- (0 to numWords -1)) {
        var s = singletonWordsPerTag(tag)+ 1e-100
        var x = (wordTagCount(word, tag) + s*(wordTagCount(word).sum + 1)/(tokenCount + numWords + 1).toDouble)/(s + tagCount(tag).toDouble)
        logPrWordGivenTag(word, tag) = math.log(x)
      }
    }
    def possibleTags(token: Int) = {
      (0 to numTags-1).filter(x =>
        if(token< numWordsTraining)
          wordTagCount(token, x)>0
        else x!= sentenceSepTag)
    }
  }


/*
Claims.
Correctly updates the following:
 wordIntMap, tagIntMap.
 numWordsTraining
 tagCount, wordTagCount, singletonWordsPerTag.
 logPrTagGivenTag
 logPrNovelWord
*/
//  Confidence in correctness: High.
//  Reason: Well tested.
  def train(iter: Iterator[Array[String]]) = {
    wordTagStatsFinal.updateCounts(iter)
//    println(logPrTagGivenTag.toString)
//    println(logPrWordGivenTag.toString)
  }
  
//  Confidence in correctness: High.
//  Reason: Well tested.
  def test(testDataIn: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
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
        tag <- wordTagStatsFinal.possibleTags(token)
    }{
      var logPrW = logPrNovelWord(tag)
      if(token < numWordsTraining)
        logPrW = logPrWordGivenTag(token, tag)
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
