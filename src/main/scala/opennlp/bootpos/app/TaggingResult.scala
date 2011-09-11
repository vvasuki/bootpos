package opennlp.bootpos.app

import scala.collection.mutable.ArrayBuffer

class TaggingResult {
  var numTestTokensKnown = 0
  var numTestTokensSeen = 0
  var numTestTokensNovel = 0
  var correctTaggingsKnown = 0
  var correctTaggingsSeen = 0
  var correctTaggingsNovel = 0

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
  def processTaggingResults(results: ArrayBuffer[Array[Boolean]], testData: ArrayBuffer[Array[String]], sentenceSepWord: String)= {
    val bUntaggedTextUsed = results(0).length>2
    // TODO: record tag Mistakes

    for {i <- testData.indices.iterator
      if(testData(i)(0) != sentenceSepWord)
    }{
      val tag = testData(i)(0);
      val bCorrect  = results(i)(0)

      // The token has been seen in tagged text.
      val bNovelToken = results(i)(1)

      // The token has been seen in untagged text - so not entirely novel.
      val bSeenToken = if(bUntaggedTextUsed) results(i)(2) else false
      update(bCorrect, bNovelToken, bSeenToken)
    }
  }
}

