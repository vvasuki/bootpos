package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import scala.collection.mutable.HashMap
import scala.collection.immutable.Set
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._

class WordTagStats(TAGNUM_IN: Int, WORDNUM_IN: Int) extends Serializable{
// The following are Double arrays because in case of EM-HMM, counts could be a non-integer.
//   Purpose: To estimate Pr(tag(i)|tag(i-1))
  val tagBeforeTagCount = new MatrixBufferDense[Double](TAGNUM_IN, TAGNUM_IN)

/*  Purpose: To estimate Pr(word)
  Property note: It is possible that wordCount.sum is larger than tagCount.sum
  due to the presence of untagged data.*/
  val wordCount = new ExpandingArray[Double](WORDNUM_IN)

//   Purpose: To estimate Pr(word|tag)
  val wordTagCount = new MatrixBufferDense[Double](WORDNUM_IN, TAGNUM_IN)
  val singletonWordsPerTag = new ExpandingArray[Double](TAGNUM_IN)
//   Property: tagCount(tag) = wordTagCount(:, tag)
  val tagCount = new ExpandingArray[Double](TAGNUM_IN)

  def numWords = wordCount.length
  def numTags = tagCount.length;

/*
  Claim: Increases table sizes appropriately.
  Confidence: High
  Reason: Proved correct.
*/
  def prepareTableSizes(numWords: Int, numTags: Int) = {
    wordTagCount.updateSize(numWords, numTags)
    singletonWordsPerTag.padTill(numTags)
    tagBeforeTagCount.updateSize(numTags, numTags)
    tagCount.padTill(numTags)
    wordCount.padTill(numWords)
  }

/*
  Assumption: The text does not contain too many empty sentences!
  Claim: wordCount updated correctly using text.
    logPrWordGivenTag recomputed.
  Confidence: High
  Reason: Proved correct.
*/
  def updateWordCount(text: ArrayBuffer[Int], hmm: HMM) = {
    text.indices.foreach(x => wordCount.addAt(x, 1))
    setLogPrWordGivenTag(hmm)
  }
/*
  Updates wordTagCount, tagCount, singleton-counts to ensure consistency.
  Confidence: High
  Reason: Well tested.
*/
  def incrementWordTagCounts(word: Int, tag: Int) = {
    wordTagCount.increment(word, tag)
    wordTagCount(word, tag) match {
      case 1 => {singletonWordsPerTag.addAt(tag, 1)}
      case 2 => {singletonWordsPerTag.addAt(tag, -1)}
      case _ => {}
    }
    wordCount.addAt(word, 1)
    tagCount.addAt(tag, 1)
  }

  /*
  Claims.
  Correctly updates the following:
  numWordsTraining
  tagCount, wordCount, wordTagCount.
  */
  //  Confidence in correctness: High.
  //  Reason: Well tested.
  def updateWordTagCount(lstData: List[Array[Int]]) = {
    for(fields <- lstData;
      tag = fields(1);
      word = fields(0)){
      incrementWordTagCounts(word, tag)
//      println(prevTag+ " t " + tag + " w "+ word)
    }
  }

  /*
  Claims.
  Correctly updates the following:
  numWordsTraining
  tagCount, wordCount, wordTagCount.
  tagBeforeTagCount
  */
  //  Confidence in correctness: High.
  //  Reason: Well tested.
  def updateCounts(lstData: List[Array[Int]], hmm: HMM) = {
    val sentenceSepTag = hmm.sentenceSepTag
    val sentenceSepWord = hmm.sentenceSepWord
    var prevTag = sentenceSepTag
    updateWordTagCount(lstData)
    for(fields <- lstData;
      tag = fields(1);
      word = fields(0)){
      tagBeforeTagCount.increment(prevTag, tag)
      prevTag = tag
    }
    
    val x = tagBeforeTagCount(sentenceSepTag, sentenceSepTag)
    if(x>0){
      tagBeforeTagCount(sentenceSepTag, sentenceSepTag) = 0
      tagCount(sentenceSepTag) = tagCount(sentenceSepTag) - x
      wordTagCount(sentenceSepWord, sentenceSepTag) = wordTagCount(sentenceSepWord, sentenceSepTag) - x
      wordCount(sentenceSepWord) = wordCount(sentenceSepWord) - x
    }
    
    setLogPrTagGivenTag(hmm)
    setLogPrWordGivenTag(hmm)
  }

  //  Confidence in correctness: High.
  //  Reason: Well tested.
  def setLogPrTagGivenTag(hmm: HMM) = {
    val numTokens = tagCount.sum
    for(tag1 <- (0 to numTags-1); tag2 <- (0 to numTags-1)) {
      var s = tagBeforeTagCount(tag2).count(x => x==1) + 1e-100
      var x = (tagBeforeTagCount(tag2, tag1) + s*tagCount(tag1)/numTokens.toDouble)/(tagBeforeTagCount(tag2).sum + s).toDouble
      hmm.logPrTagGivenTag(tag1, tag2) = math.log(x)
//       println(tag1 + "|" + tag2+ " = " + x)
    }
  }

//  Confidence in correctness: High.
//  Reason: Proved Correct.
  def setLogPrWordGivenTag(hmm: HMM) = {
//     First, calculate Pr(word)
/*    NOte: It is possible that wordCount.sum is larger than tagCount.sum
    due to the presence of untagged data.
    Below we ensure that we take advantage of untagged data.*/
    val numTokens = wordCount.sum
    // Pr(word) independent of tags, with add 1 smoothing.
    val prWord = wordCount.map(x => (x + 1)/(numTokens + numWords + 1).toDouble)

//     Pr(word|tag), smoothed using singleton count and Pr(word)
    for(tag <- (0 to numTags-1); word <- (0 to numWords -1)) {
      val s = singletonWordsPerTag(tag)+ 1e-100
      var x = s*prWord(word)
      if(word < wordTagCount.length)
        x = x + wordTagCount(word, tag)
      x = x/(s + tagCount(tag).toDouble)
      hmm.logPrWordGivenTag(word, tag) = math.log(x)
    }

    //  Confidence in correctness: High.
    //  Reason: Well tested.
    for(tag<- (0 to numTags-1)) {
      var s = singletonWordsPerTag(tag)+ 1e-100
      var x = (s/(numTokens + numWords + 1).toDouble)/(s + tagCount(tag).toDouble)
      hmm.logPrNovelWord(tag) = math.log(x)
    }

  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def possibleTags(token: Int, hmm: HMM) = {
    (0 to numTags-1).filter(x =>
      if(token< hmm.numWordsTraining)
        wordTagCount(token, x)>0
      else x!= hmm.sentenceSepTag)
  }

  override def toString = {
    var str = ("tagC " + tagCount)
//     str +=("\n tagBefTag.rowSum " + tagBeforeTagCount.matrix.map(_.sum))
//     str +=("\n tagBefTag " + tagBeforeTagCount) \
//     str +=("\n wrdTag " + wordTagCount)
    str
  }


}

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
  var logPrWordGivenTag = new MatrixBufferDense[Double](WORDNUM_IN, TAGNUM_IN)
  var logPrNovelWord = new ExpandingArray[Double](TAGNUM_IN)
  var numWordsTraining = 0

  val wordTagStatsFinal = new WordTagStats(TAGNUM_IN, WORDNUM_IN)
  val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)



//   Confidence: High.
//   Reason: Proved correct.
  def getPrWordGivenTag(word: Int, tag: Int) = {
    if(word < logPrWordGivenTag.length) logPrWordGivenTag(word, tag)
    else logPrNovelWord(tag)
  }

//   Confidence: High.
//   Reason: Proved correct.
  def getArcPr(tag:Int, prevTag: Int, word: Int) = {
    logPrTagGivenTag(tag, prevTag) + getPrWordGivenTag(word, tag)
  }

//   Confidence: High.
//   Reason: Proved correct.
  override def toString = {
    val randWord = (math.random * logPrWordGivenTag.length).toInt
    val randTag = (math.random * logPrTagGivenTag.length).toInt
    var str = "t="+randTag + " w="+randWord
    str +=("\nT|T " + logPrTagGivenTag.map(math.exp))
    str +=("\nW|T " + logPrWordGivenTag(randWord).map(math.exp))
//     str +=("NW|T " + logPrNovelWord.map(math.exp))
    str +=("\nT|T=t " + logPrTagGivenTag.getCol(randTag).map(math.exp).sum)
    str += "\n sum T|T=t" + math.exp(logPrTagGivenTag.colFold(math.log(0))(randTag, mathUtil.logAdd))
    str += "\n sum W|T=t" + mathUtil.logAdd(logPrWordGivenTag.colFold(math.log(0))(randTag, mathUtil.logAdd), logPrNovelWord(randTag))
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
    println(wordTagStatsFinal)
    println(this)
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
//  Confidence in correctness: Low.
//  Reason: Implementation incomplete.
  override def trainWithDictionary(iter: Iterator[Array[String]], wordSet: Set[String] = null) = {
    val lstData = iter.filter(x => if(wordSet != null)wordSet.contains(x(0)) else true).
      map(x => Array(getWordId(x(0)), getTagId(x(1)))).toList
    wordTagStatsFinal.updateWordTagCount(lstData)
    logPrTagGivenTag = new MatrixBufferDense[Double](numTags, numTags, 1/numTags, true)
    wordTagStatsFinal.setLogPrWordGivenTag(this)
    wordTagStatsFinal.incrementWordTagCounts(sentenceSepWord, sentenceSepTag)
    numWordsTraining = numWordsTotal
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
