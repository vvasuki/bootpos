package opennlp.bootpos.app

import scala.collection.mutable.ArrayBuffer
import scala.collection.IndexedSeq
import org.slf4j.LoggerFactory
import opennlp.bootpos.tag._

class TaggingResult {
  val log = LoggerFactory.getLogger(this.getClass)
  var numTestTokensKnown = 0
  var numTestTokensSeen = 0
  var numTestTokensNovel = 0
  var correctTaggingsKnown = 0
  var correctTaggingsSeen = 0
  var correctTaggingsNovel = 0
  var maxErrorTags = ""

  var accuracy = 0.0
  var accuracyKnown = 0.0
  var accuracyNovel = 0.0
  var novelTokensFrac = 0.0

//  Confidence in correctness: High.
//  Reason: Well tested.
  def update(bCorrect: Boolean, bNovelToken: Boolean, bSeenToken: Boolean) = {
    if(!bNovelToken) {
      if(bCorrect) correctTaggingsKnown  = correctTaggingsKnown  + 1
      numTestTokensKnown = numTestTokensKnown + 1
    }
    else if(bSeenToken){
      if(bCorrect) correctTaggingsSeen  = correctTaggingsSeen  + 1
      numTestTokensSeen = numTestTokensSeen + 1
    }
    else {
      if(bCorrect) correctTaggingsNovel  = correctTaggingsNovel  + 1
      numTestTokensNovel = numTestTokensNovel + 1
    }
  }

  def toTsv = {
    accuracy + "\t" + accuracyKnown + "\t" + accuracyNovel + "\t" + novelTokensFrac
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def updateAccuracy = {
    var correctTaggings = correctTaggingsKnown + correctTaggingsNovel
    var numTestTokens = numTestTokensKnown + numTestTokensNovel
    accuracy = correctTaggings/ numTestTokens.toDouble
    accuracyKnown = correctTaggingsKnown/ numTestTokensKnown.toDouble
    accuracyNovel = correctTaggingsNovel/ numTestTokensNovel.toDouble
    novelTokensFrac = numTestTokensNovel/numTestTokens.toDouble

    printf("Accuracy: %.3f, (Known: %.3f, Novel: %.3f)\n", accuracy, accuracyKnown, accuracyNovel)
    printf("Non training tokens: %.3f\n", novelTokensFrac)
  }


//  Confidence in correctness: High.
//  Reason: Well tested.
  def processTaggingResults(bCorrect: IndexedSeq[Boolean], bNonTraining: IndexedSeq[Boolean], bUnseen: IndexedSeq[Boolean] = null, bSentenceSep: IndexedSeq[Boolean])= {
    val bUntaggedTextUsed = bUnseen == null

    bCorrect.indices.filterNot(bSentenceSep(_)).foreach(i => {
      val bSeenToken = if(bUntaggedTextUsed) bNonTraining(i) && !bUnseen(i)
        else false
      update(bCorrect(i), bNonTraining(i), bSeenToken)
    })
  }
}

class TaggerTester(tagger: Tagger) {
  val log = LoggerFactory.getLogger(this.getClass)
  val intMap = tagger.intMap

//   ASSUMPTIONS: tokens and tags are assumed to be in the correct case.
  def test(testData: ArrayBuffer[Array[String]]) = {
    val tokens = testData.map(_(0))
    val tagsActual = testData.map(_(1))
    val tags = tagger.tag(tokens)
    
    val bTokenNonTraining = tokens.map(intMap.isNonTraining)
    val bTokenUnseen = tokens.map(intMap.isUnseen)
    val bSentenceSep = tokens.map(_ == intMap.sentenceSepWordStr)
    val bCorrect = tags map (_ == tagsActual)

    val tagResults = new TaggingResult()
    tagResults.maxErrorTags = examineMistakes(tags, tagsActual).toString
    tagResults.processTaggingResults(bCorrect, bTokenNonTraining, bTokenUnseen, bSentenceSep)
    tagResults.updateAccuracy
    tagResults

  }
  
  def examineMistakes(tags: IndexedSeq[String], tagsActual: IndexedSeq[String]) = {
    val taggingErrors = tags zip tagsActual filterNot (x => x._1 == x._2) groupBy (x => x) mapValues (_.size)
    val maxError = taggingErrors.values.max
    taggingErrors.filter(x => x._2 == maxError).keys.head
  }
}