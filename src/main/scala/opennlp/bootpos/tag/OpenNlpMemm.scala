package opennlp.bootpos.tag

import opennlp.bootpos.util.io._
import opennlp.tools.postag._
import opennlp.tools.util._
import opennlp.tools.util.model._
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.immutable.Stream

class OpenNLP(languageCode: String, sentenceSepTagStr :String, sentenceSepWordStr: String) extends Tagger{
  val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)


  var model: POSModel = null
  
// Computation:
//  Update wordIntMap.
//  Train the POSModel.
//  Confidence in correctness: High.
//  Reason: Proved Correct.
//  Claims:
//    wordIntMap is updated correctly.
//    model is set appropriately.
  def train(wordTagIter: Iterator[Array[String]]) = {
  //    update wordIntMap appropriately.
    val wordTagLst = wordTagIter.toList
    wordTagLst.foreach(x => getWordId(x(0).map(_.toUpper))) // update wordIntMap.

    val sentenceStream = new TaggedSentenceStream(wordTagLst.iterator, sentenceSepTagStr)
    val numIters = 100; val eventCutoff = 5
    model = POSTaggerME.train(languageCode.map(_.toLower), sentenceStream, ModelType.MAXENT,
      null, null, eventCutoff, numIters);
  }
  
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def test(testData: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
    val numWords  = wordIntMap.size
    val numTokens = testData.length
    var resultPair = new ArrayBuffer[Array[Boolean]](numTokens + 1)

    val tagger = new POSTaggerME(model);
  //    update wordIntMap appropriately.
    testData.foreach(x => getWordId(x(0).map(_.toUpper))) // update wordIntMap.

    var sentenceStream = new TaggedSentenceStream(testData.iterator, sentenceSepTagStr)

    var sample = sentenceStream.read
    while(sample != null){
      var tokens = sample.getSentence
      var tagsActual = sample.getTags
      val tagsPredicted = tagger.tag(tokens)
      val sentenceLength = tokens.length
      tokens.indices.foreach(i => {
        val bNovel = getWordId(tokens(i).map(_.toUpper)) >= numWords
        val bCorrect = tagsActual(i) == tagsPredicted(i)
        resultPair += Array(bCorrect, bNovel)
        })
      // Add a result corresponding to correct tagging of sentence-separator word.
      if(1 <= testData.length - resultPair.length )
        resultPair += Array(true, false)
      sample = sentenceStream.read
    }

    // openNLP's Evaluator
    // Confidence: High
    // Reason: Proved correct.
    // Observed to yield same result as above.
    def openNLPEval = {
      sentenceStream = new TaggedSentenceStream(testData.iterator, sentenceSepTagStr)
      val evaluator =
          new POSEvaluator(new opennlp.tools.postag.POSTaggerME(model));
      System.out.print("Evaluating ... ");
      evaluator.evaluate(sentenceStream);
      System.out.println("Accuracy: " + evaluator.getWordAccuracy());
    }

    resultPair
  }

}
