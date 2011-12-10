package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import scala.collection.mutable.HashMap
import scala.collection.immutable.Set
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._

class HMMTagger extends Tagger{
//   Probabilities are stored in log space to avoid underflow.
// logPrTGivenT, due to its small size, should be precomputed.
  var logPrTGivenT = new MatrixBufferDense[Double](intMap.TAGNUM_IN, intMap.TAGNUM_IN)
/*
  Considerations in deciding whether to compute logPrWGivenT:
  1. Test set may not contain many words seen in training set.
  2. We may want to avoid repeated computation for words which appear multiple times.
  My guess is that [2] outweighs [1].
*/
  var logPrWGivenT = new MatrixBufferDense[Double](intMap.WORDNUM_IN, intMap.TAGNUM_IN, defaultValue = math.log(0))
  var logPrNovelWord = new ExpandingArray[Double](intMap.TAGNUM_IN, defaultValue = math.log(0))


//   Confidence: High.
//   Reason: Proved correct.
  def getPrWGivenT(word: Int, tag: Int) = {
    if(word < logPrWGivenT.numRows) logPrWGivenT(word, tag)
    else logPrNovelWord(tag)
  }

//   Confidence: High.
//   Reason: Proved correct.
  def getArcPr(tag:Int, prevTag: Int, word: Int) = {
    logPrTGivenT(tag, prevTag) + getPrWGivenT(word, tag)
  }

//   Confidence: High.
//   Reason: Proved correct.
  def checkLogPrWGivenT(tag: Int) = {
    mathUtil.logAdd(logPrWGivenT.colFold(math.log(0))(tag, mathUtil.logAdd), logPrNovelWord(tag))
  }

//   Confidence: High.
//   Reason: Proved correct.
  override def toString = {
    val randWord = (math.random * logPrWGivenT.numRows).toInt
    val randTag = (math.random * logPrTGivenT.numRows).toInt
    var str = "hmm:"
    str += "\nt="+randTag + " w="+randWord
//     str +=("\nT|T " + logPrTGivenT.map(math.exp))
//    str +=("\nW=w|T " + logPrWGivenT(randWord).map(math.exp))

    val prTGivenTsums = (0 to numTags-1).map(
      logPrTGivenT.colFold(math.log(0))(_, mathUtil.logAdd))
    str += "\nBig sum T|T " +prTGivenTsums.indices.filter(x => math.abs(prTGivenTsums(x))> 1E-4)
//     str += "\n " + prTGivenTsums
//     str += "\n " + logPrTGivenT.getCol(0).map(math.exp)
//     str += "\n " + logPrTGivenT.getCol(9).map(math.exp)

//     str += "\n sum W|T=t " + checkLogPrWGivenT(randTag)
//     str +=("\nNW|T " + logPrNovelWord)

    str +=("\nBig sum_W Pr(W|T) " + (0 to numTags-1).
      map(checkLogPrWGivenT(_)).filter(math.abs(_)> 1E-4))
    str +=("\n W|T=### "+ logPrWGivenT.getCol(intMap.sentenceSepTag).filter(_ != Double.NegativeInfinity))
    str +=("\n NW|T=### "+ logPrNovelWord(intMap.sentenceSepTag))
    str += "\n Pr(NW|T) " + logPrNovelWord
    str += "\n===\n"
    str
  }


  
//  Confidence in correctness: High.
//  Reason: Well tested.
  def tag(testDataIn: ArrayBuffer[String])= {
    log info "tagging"
    log info(this.toString)
    val testData = testDataIn.map(intMap.getWordId)
    val numTokens = testData.length
    val numTags = intMap.numTags;

    var bestPrevTag = new MatrixBufferDense[Int](numTokens + 1, numTags)
    var logPrSequence = new MatrixBufferDense[Double](numTokens + 1, numTags, defaultValue=math.log(0))

    // A flag to indicate that we are not expecting the next token
    // to be sentenceSepWordStr. Necessary for correct perplexity calculation.
    var bSeekSentence = true

    log debug intMap.possibleTags(intMap.sentenceSepTag) + " tag for  ###"

    logPrSequence(0, intMap.sentenceSepTag) = math.log(1)
    for{tokenNum <- 1 to numTokens;
        token = testData(tokenNum-1)
        tag <- intMap.possibleTags(token)
    }{
      val logPrW = getPrWGivenT(token, tag)
      var logPrJ = matrixMath.vp(logPrSequence(tokenNum-1), logPrW)
//      Ensure that perplexity is not affected by empty sentences.
      if(!(bSeekSentence && token==intMap.sentenceSepWord))
        logPrJ = matrixMath.vp(logPrJ, logPrTGivenT(tag))
      
      logPrSequence(tokenNum, tag) = logPrJ.max
      bestPrevTag(tokenNum, tag) = logPrJ.indexOf(logPrSequence(tokenNum, tag))

      bSeekSentence = token == intMap.sentenceSepWord

//      log info("logPrSeq "+ logPrSequence(tokenNum))
//      log info("# "+tokenNum + " w " + token + " tg "+ tag + " tg_{-1} "+ bestPrevTag(tokenNum, tag))
    }

    val bestTags = new ExpandingArray[Int](numTokens)
    bestTags(numTokens-1) = logPrSequence(numTokens).indexOf(logPrSequence(numTokens).max)
    var perplexity = math.exp(-logPrSequence(numTokens, bestTags(numTokens-1))/numTokens)
    log info("Perplexity: " + perplexity)

    for(tokenNum <- numTokens-2 to 0 by -1) {
      bestTags(tokenNum) = bestPrevTag(tokenNum+2, bestTags(tokenNum+1))
    }
    
//      log info(tokenNum + " : " + token + " : "+ resultPair(tokenNum))
    // log debug testData.mkString(" ")
    // log debug bestTags.mkString(" ")
    bestTags.map(intMap.getTagStr)
  }

}

class HMMTrainer(sentenceSepTagStr :String, sentenceSepWordStr: String) extends TaggerTrainer(sentenceSepTagStr, sentenceSepWordStr) {
  override val tagger = new HMMTagger
  setIntMap
  val wordTagStatsFinal = new WordTagStats(intMap.TAGNUM_IN, intMap.WORDNUM_IN, intMap)

/*
Claims.
Correctly updates the following:
  wordIntMap, tagIntMap.
  numWordsTraining
  tagCount, wordTagCount, singletonWordsPerTag.
  logPrTGivenT
  logPrNovelWord
  logPrWGivenT
*/
//  Confidence in correctness: High.
//  Reason: Well tested.
  def train(iter: Iterator[Array[String]]) = {
    val lstData = iter.map(x => Array(intMap.getWordId(x(0)), intMap.getTagId(x(1)))).toList
    log info "Training using tagged sequence."
//     print(lstData.take(10).map(_.mkString(":")).mkString(", "))
    intMap.updateWordTagList(lstData)

    wordTagStatsFinal.updateCounts(lstData, tagger)
/*    log info(wordTagStatsFinal)
    log info(tagger)*/
    tagger
  }

/*
Claims.
Correctly updates the following:
  wordIntMap, tagIntMap.
  tagCount, wordTagCount.
  logPrTGivenT
  logPrNovelWord
  logPrWGivenT

Problems to consider when creating an initial HMM model from a dictionary.
1] The dictionary may be incomplete in two ways:
  1a] There may be missing words.
  1b] Even when a word is included, some potential parts of speech may not be mentioned.
2] We have no word-sequence and tag-sequence information.

Model parameters:
  Pr(t_i|t_{i-1}) = Pr(t_i) - or perhaps just 1/numTags.
  The latter choice may be preferable because it may
  facilitate better learning of Pr(t_i|t_{i-1}) during EM.

  Model Pr(w \notin W) arbitrarily - or learn it using the dictionary and raw data.
  For details about Pr(w|t) initialization, see comments for setLogPrWGivenT.
  ALTERNATIVE: Use information from singletonWordsPerTag.

NOTE regarding singletonWordsPerTag:
  This proceudure updates singletonWordsPerTag,
    and scales it down appropriately.
  That data is not used in setting Pr(w|t),
    but is used in later EM iterations - albeit in an attenuated sense.
  This may be justified by saying that tags with greater words associated with them in a dictionary are less likely to be 'closed' to novel words.

  An ALTERNATIVE to the setting is to do the following:
    Set it to 1-dictionary.completeness.
    But this alternative seems worse as it does not distinguish between tags.


NOTES:
  Tag count should be updated during HMM.
  Ensure EM iterations start with fresh counts when starting point has been deduced from a wiktionary.
*/
//  Confidence in correctness: Medium.
//  Reason: Seems to be fine.
  override def trainWithDictionary(dictionary: Dictionary) = {
    var lstData = dictionary.lstData.
    map(x => Array(intMap.getWordId(x(0)), intMap.getTagId(x(1))))
    intMap.updateWordTagList(lstData)

    wordTagStatsFinal.updateWordTagCount(lstData.toList, bUpdateWordCount = false)
    val bUniformModelForTags = true
    if(bUniformModelForTags)
      tagger.logPrTGivenT = new MatrixBufferDense[Double](numTags, numTags, math.log(1/numTags.toDouble), true)
    else
      wordTagStatsFinal.setLogPrTGivenTFromTCount(tagger)
    wordTagStatsFinal.setLogPrWGivenT(tagger, dictionary)

    wordTagStatsFinal.singletonWordsPerTag = wordTagStatsFinal.singletonWordsPerTag.map(_ / lstData.length.toDouble)
    log info("tokens in dictionary data: " + lstData.length)
    log info(wordTagStatsFinal.toString)
    log info(tagger.toString)
    tagger
  }
}
