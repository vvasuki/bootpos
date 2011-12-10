package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import scala.collection.mutable.HashMap
import scala.collection.immutable.Set
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._
import org.slf4j.LoggerFactory

class WordTagStats(TAGNUM_IN: Int, WORDNUM_IN: Int, intMap: IntRepresentor) extends Serializable{
  val log = LoggerFactory.getLogger(this.getClass)
// entry (i, j) will count number of occurances of tag i before tag j
// The following are Double arrays because in case of EM-HMM, counts could be a non-integer.
//   Purpose: To estimate Pr(tag(i)|tag(i-1))
  var tagBeforeTagCount = new MatrixBufferDense[Double](TAGNUM_IN, TAGNUM_IN)

/*  Purpose: To estimate Pr(word)
  Property note: It is possible that wordCount.sum > tagCount.sum 
  due to the presence of untagged data.
  */
  var wordCount = new ExpandingArray[Double](WORDNUM_IN)

//   Purpose: To estimate Pr(word|tag)
  var wordTagCount = new MatrixBufferDense[Double](WORDNUM_IN, TAGNUM_IN)
  var singletonWordsPerTag = new ExpandingArray[Double](TAGNUM_IN)
//   Property: tagCount(tag) = wordTagCount(:, tag)
  var tagCount = new ExpandingArray[Double](TAGNUM_IN)

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
  Scale down nearly everything updated with updateCount method:
    wordTagCount, singletonWordsPerTag,
    tagBeforeTagCount, tagCount.
  Do not scale down the following:
    wordCount - that is used to compute word probability.
    singletonWordsPerTag - that should be scaled as necessary separately.
  */
  def scaleDown(p: Double) = {
    wordTagCount = wordTagCount.map(_ * p)
    tagBeforeTagCount = tagBeforeTagCount.map(_ * p)
    tagCount = tagCount.map(_ * p)
    // log info(this)
  }

/*
  Assumption: The text does not contain too many empty sentences!
  Claim: wordCount updated correctly using text.
    logPrWGivenT recomputed.
  Confidence: High
  Reason: Proved correct.
*/
  def updateWordCount(text: ArrayBuffer[Int], hmm: HMMTagger, bUpdateEmissionProb: Boolean = true) = {
    log info "updating wordCount with bUpdateEmissionProb "+ bUpdateEmissionProb 
    text.indices.foreach(x => wordCount.addAt(x, 1))
    if(bUpdateEmissionProb ) setLogPrWGivenT(hmm)
  }
/*
  Updates wordTagCount, tagCount, singleton-counts to ensure consistency.
  wordCount updated only if bUpdateWordCount == true.
  Confidence: High
  Reason: Well tested.
*/
  def incrementWordTagCounts(word: Int, tag: Int, bUpdateWordCount: Boolean = true) = {
    wordTagCount.increment(word, tag)
    wordTagCount(word, tag) match {
      case 1 => {singletonWordsPerTag.addAt(tag, 1)}
      case 2 => {singletonWordsPerTag.addAt(tag, -1)}
      case _ => {}
    }
    tagCount.addAt(tag, 1)
    if(bUpdateWordCount)
      wordCount.addAt(word, 1)
    else
      wordCount.padTill(word)
  }

  /*
  Claims.
  Correctly updates the following:
  tagCount, wordCount, wordTagCount.
  */
  //  Confidence in correctness: High.
  //  Reason: Well tested.
  def updateWordTagCount(lstData: List[Array[Int]], bUpdateWordCount: Boolean = true) = {
    for(fields <- lstData;
      tag = fields(1);
      word = fields(0)){
      incrementWordTagCounts(word, tag, bUpdateWordCount)
//      log info(prevTag+ " t " + tag + " w "+ word)
    }
  }

  /*
  Claims.
  Correctly updates the following:
  tagCount, wordCount, wordTagCount, singletonWordsPerTag.
  tagBeforeTagCount.
  pr(W|T) and Pr(T|T) are then updated.
  */
  //  Confidence in correctness: High.
  //  Reason: Well tested.
  def updateCounts(lstData: List[Array[Int]], hmm: HMMTagger) = {
    val sentenceSepTag = intMap.sentenceSepTag
    val sentenceSepWord = intMap.sentenceSepWord
    var prevTag = sentenceSepTag
    updateWordTagCount(lstData)
    for(fields <- lstData;
      tag = fields(1);
      word = fields(0)){
      tagBeforeTagCount.increment(prevTag, tag)
      prevTag = tag
    }

// Doing the below adjustment may lead to errors, and it is mostly unimportant anyway.
//     val x = tagBeforeTagCount(sentenceSepTag, sentenceSepTag)
//     if(x>0){
//       tagBeforeTagCount(sentenceSepTag, sentenceSepTag) = 0
//       tagCount(sentenceSepTag) = tagCount(sentenceSepTag) - x
//       wordTagCount(sentenceSepWord, sentenceSepTag) = wordTagCount(sentenceSepWord, sentenceSepTag) - x
//       wordCount(sentenceSepWord) = wordCount(sentenceSepWord) - x
//     }

    setLogPrTGivenT(hmm)
    setLogPrWGivenT(hmm)
  }

//  Prerequisites:
//     Set tagBeforeTagCount, tagCount.
//       If tagBeforeTagCount contains integers, it is used for smoothing.
//  Confidence in correctness: High.
//  Reason: Well tested.
  def setLogPrTGivenT(hmm: HMMTagger) = {
    val numTokens = tagCount.sum
    for(tag1 <- (0 to numTags-1); tag2 <- (0 to numTags-1)) {
      var s = tagBeforeTagCount(tag2).count(x => x==1) + 1e-100
      val backOffProb = tagCount(tag1)/numTokens.toDouble
      var x = (tagBeforeTagCount(tag2, tag1) + s*backOffProb)/(tagBeforeTagCount(tag2).sum + s).toDouble
      hmm.logPrTGivenT(tag1, tag2) = math.log(x)
//       log info(tag1 + "|" + tag2+ " = " + x)
    }
  }

  // Set Pr(t_i|t_{i-1}) = Pr(t_i).
  def setLogPrTGivenTFromTCount(hmm: HMMTagger) = {
    val numTokens = tagCount.sum
    for(tag1 <- (0 to numTags-1); tag2 <- (0 to numTags-1)) {
      var x = tagCount(tag1)/numTokens
      hmm.logPrTGivenT(tag1, tag2) = math.log(x)
//       log info(tag1 + "|" + tag2+ " = " + x)
    }
  }

// ASSUMPTION: A 'seen word' cannot have an unobserved connection to a tag.
// Ensure that there is no smoothing for sentenceSepTag.
//   Prerequisites:
//     Set wordCount, wordTagCount, singletonWordsPerTag.
//  Confidence in correctness: High.
//  Reason: Proved Correct.
  def setLogPrWGivenT(hmm: HMMTagger) = {
//     First, calculate Pr(word)
/*    NOte: It is possible that wordCount.sum is larger than tagCount.sum
    due to the presence of untagged data.
    Below we ensure that we take advantage of untagged data.*/
    val numTokens = wordCount.sum
    // Pr(word) independent of tags, with add 1 smoothing.
    val prWord = wordCount.map(x => (x + 1)/(numTokens + numWords + 1).toDouble)
    val sentenceSepTag = intMap.sentenceSepTag

//     Pr(word|tag), smoothed using singleton count and Pr(word)
//     Note that initially wordTagCount.numRows < wordCount.length is possible.
//     But this is corrected after the below loop.
    for(tag <- (0 to numTags-1).filterNot(_ == sentenceSepTag); word <- (0 to numWords -1)) {
      val s = singletonWordsPerTag(tag)+ 1e-100
      var x = 0.0

      if(word<wordTagCount.numRows && wordTagCount(word, tag) > 0)
        x = wordTagCount(word, tag) + s*prWord(word)
      else
        x = s*prWord(word)
      x = x/(s + tagCount(tag).toDouble)
      hmm.logPrWGivenT(word, tag) = math.log(x)
    }
    hmm.logPrWGivenT(intMap.sentenceSepWord, sentenceSepTag) = math.log(1)

    //  Confidence in correctness: High.
    //  Reason: Well tested.
    for(tag<- (0 to numTags-1).filterNot(_ == sentenceSepTag)) {
      var s = singletonWordsPerTag(tag)+ 1e-100
      var x = (s/(numTokens + numWords + 1).toDouble)/(s + tagCount(tag).toDouble)
      hmm.logPrNovelWord(tag) = math.log(x)
    }
    hmm.logPrNovelWord(sentenceSepTag) = math.log(0)

  }

/*
  ASSUMPTION: dictionary.completeness < 1.
Let W be the set of words in the dictionary.
Let W_t be the set of words with tag t according to the tag-dictionary.

Probability of seeing a novel word, given a tag is set to be a constant:
Pr(w \notin W_t | t) = Pr(w \notin W) = 1 - dictionary.completeness.

Probability of seeing an particular in-dictionary word w given that tag is t is modeled by the uniform distribution.
Pr(w | w \in W_t, t) = 1/|W_t|

Thus, the probability of seeing a certain in-dictionary word w given tag t is:
Pr(w AND w \in W_t | t) = Pr(w | w \in W_t, t) Pr(w \in W_t | t) = (1/|W_t|) Pr(w \in W_t | t)

TODO:
  Smoothen this to allow small probability for
  unobserved word tag associations. This may be necessary in case we use a very incomplete dictionary.
  Confidence: High
  Reason: Well tested.
*/
  def setLogPrWGivenT(hmm: HMMTagger, dict: Dictionary) = {
    log info "Setting pr(w|t) fromm dictionary."
    hmm.logPrNovelWord.padTill(numTags, math.log(1 - dict.completeness))
    val sentenceSepTag = intMap.sentenceSepTag
    hmm.logPrNovelWord(sentenceSepTag) = math.log(0)
    
    val numTaggedWords = wordTagCount.numRows
    for(word <- (0 to numTaggedWords-1); tag<- (0 to numTags-1).filterNot(_ == sentenceSepTag)){
      var wtCnt= wordTagCount(word, tag)
      var x = (1 - math.exp(hmm.logPrNovelWord(tag)))*(wtCnt)/(tagCount(tag))
      hmm.logPrWGivenT(word, tag) = math.log(x)
    }
    hmm.logPrWGivenT(intMap.sentenceSepWord, sentenceSepTag) = math.log(1)
  }



//  Confidence in correctness: High.
//  Reason: Well tested.
  override def toString = {
    val randTag = (math.random * numTags).toInt
    var str = "WTStats:"
    str += "numTags "+ numTags + " numW "+ numWords
    str += "\nt="+randTag
    str += ("\ntagC " + tagCount.mkString(" "))
    str += ("\nwtC " + wordTagCount.colSums.mkString(" "))
    str += ("\nsingC " + singletonWordsPerTag.mkString(" "))
//     str +=("\n tagBefTag.rowSum " + tagBeforeTagCount.matrix.map(_.sum))
//     str +=("\n tagBefTag " + tagBeforeTagCount) \
//     str +=("\n wrdTag " + wordTagCount)
    if(numTags != 0){
    str += ("\n tagC(t) " + tagCount(randTag))
    str += ("\n wtC(:,t) " + wordTagCount.getCol(randTag).sum)
    }
    str += "\n===\n"
    str
  }


}

class WordTagStatsProb(TAGNUM_IN: Int, WORDNUM_IN: Int, intMap: IntRepresentor) extends WordTagStats(TAGNUM_IN, WORDNUM_IN, intMap){

/*
  Example against which much of this code was verified:
    http://comp.ling.utexas.edu/_media/courses/2008/fall/natural_language_processing/eisner-icecream-forwardbackward.xls
  Purpose:
      0. Execute forward/ backward algorithm.
      1. Update wordTagCount, tagBeforeTagCount, tagCount
      2. Update: logPrTGivenT (not smoothed)
          logPrWGivenT logPrNovelWord (smoothed using singletonWordsPerTag)
  Assumptions:
    1. singletonWordsPerTag is to be updated before calling this, if necessary.
      As explained in http://comp.ling.utexas.edu/_media/courses/2010/spring/natural_language_processing/nlp10_hw4.pdf ,
        it should not be updating during this procedure.
      It is considered in computing Pr(W|T).
    2. Note that wordCount is not updated below - it is updated beforehand,
        and not once per EM iteration.
  Confidence: High.
  Reason: Proved correct. Also verified with ic test data.
    1. See comments below.
*/
  def updateCountsEM(text: ArrayBuffer[Int], hmm: HMMTagger, forwardPr: MatrixBufferDense[Double], backwardPr: MatrixBufferDense[Double]) = {
    val numTokens = text.length
    val sentenceSepTag = intMap.sentenceSepTag

    val numTokensUntagged = forwardPr.numRows

    val logPrTokens = forwardPr(numTokensUntagged-1, sentenceSepTag)
    log info("logPrTokens " + logPrTokens)

/*
    Claim: wordTagCount, tagCount correctly updated below.
    Confidence: Moderate.
    Reason: Not sure whether underflow errors occur.
      Otherwise proved correct.
*/
    prepareTableSizes(hmm.numWordsTotal, numTags)
    for{i <- 1 to numTokens-1
      token = text(i)
      tag <- intMap.possibleTags(token)
    }{
      val prTag = math.exp(forwardPr(i, tag) + backwardPr(i, tag) - logPrTokens)
      wordTagCount(token, tag) = wordTagCount(token, tag) + prTag
      tagCount(tag) = tagCount(tag) + prTag
    }
/*
    Claim: tagBeforeTagCount correctly updated below.
    Confidence: Moderate.
    Reason: Not sure whether underflow errors occur.
      Otherwise proved correct.
*/
    for{i <- 1 to numTokens-1
      token = text(i)
      tag <- intMap.possibleTags(token)
      prevTag <- intMap.possibleTags(text(i-1))
    }{
      val prTagPair = forwardPr(i-1, prevTag) + backwardPr(i, tag) - logPrTokens + hmm.getArcPr(tag, prevTag, token)
      tagBeforeTagCount(prevTag, tag) = tagBeforeTagCount(prevTag, tag) + math.exp(prTagPair)
    }

    log info(this.toString)

    setLogPrTGivenT(hmm)
    setLogPrWGivenT(hmm)
    log info(hmm.toString)
  }

/*
  Purpose:
      1. Update various counts required for computing logPrTGivenT etc..
        These include: wordTagCount, tagBeforeTagCount,
        tagCount, wordCount
      1a. Note that wordCount is not updated below
        as initial-parameter computation doesn't require it'-
        it is updated later, before the EM iteration.
      2. Update: logPrTGivenT logPrWGivenT logPrNovelWord
  ASSUMPTION: singletonWordsPerTag should be updated beforehand.
  Confidence: High.
  Reason: Proved correct.

*/
  def updateCountsPr(text: ArrayBuffer[Int], hmm: HMMTagger, prTag: IndexedSeq[IndexedSeq[(Int, Double)]]) = {
    log info "Updating counts using label distribution"
    // update wordCount
    updateWordCount(text, hmm, bUpdateEmissionProb = false)
    
    val numTokens = text.length
    text.indices.foreach(i => {
      val token = text(i)
      val tagDistribution = prTag(i)
      tagDistribution.foreach(x => {
        val tag = x._1; val probability = x._2
        // Update tagCount
        //   - the above is good mainly for being an easy way to count numTokens.
        //  Its main purpose of calculating Pr(tag), is wasted because little weight (~ 0)
        //    is likely to be given to it in finding Pr(t|t_0)
        tagCount.addAt(tag, probability)

        // Update wordTagCount
        wordTagCount.addAt(token, tag, probability)

          // Update tagBeforeTagCount
        if(i>0) prTag(i-1).foreach(y => {
          val prevTag = y._1; val probabilityPrevTag = y._2
          tagBeforeTagCount.addAt(prevTag, tag, probability*probabilityPrevTag)
        })
      })
    })
    setLogPrTGivenT(hmm)
    setLogPrWGivenT(hmm)
    log info(hmm.toString)
    
  }

}

