package opennlp.bootpos.tag

import opennlp.tools.postag._
import opennlp.tools.util._
import opennlp.tools.util.model._
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.immutable.Stream

class OpenNLP(languageCode: String, sentenceSepTagStr :String, sentenceSepWordStr: String) extends Tagger{
  val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)
  //  Warning: The implementation of reset does not behave as expected.
  //  This is because of the difficulty in constructing a stream from a scala iterator.
  //  This is by design: we think that reset() will not be called.
  class TaggedSentenceStream(wordTagIter: Iterator[Array[String]]) extends ObjectStream[POSSample] {
//    Supposed to equal one more than the maximum wordId found in wordTagIter so far by read().
//    Confidence: High
//    Reason: read() proved correct.

//  Confidence: Low.
//  Reason: Deliberately not well designed.
//  We assume that the defined funcitonality is not required.
    def close() = {println("TaggedSentenceStream.close")}

//  Confidence: Low.
//  Reason: Deliberately not well designed.
//  We assume that the defined funcitonality is not required.
//    @throws(classOf[java.io.IOException])
    def reset() = {throw new java.io.IOException("Undefined operation.")}

//  Returns:
//    the next object or null to signal that the stream is exhausted.
//  Side-effect:
//    update wordIntMap appropriately.
//  Confidence in correctness: High.
//  Reason: Proved correct.
    def read : POSSample = {
      if(wordTagIter.isEmpty) return null

      var words = new ArrayBuffer[String](20)
      var tags = new ArrayBuffer[String](20)

      wordTagIter.takeWhile(x => x(1) != sentenceSepTagStr).foreach(x => {
          getWordId(x(0).map(_.toUpper)); // update wordIntMap.
          words += x(0); tags += x(1)})
//      println(words)
//      println(tags)
      new POSSample(words.toArray, tags.toArray)
    }
  }


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
    val sentenceStream = new TaggedSentenceStream(wordTagIter)
    val numIters = 100; val eventCutoff = 5
    model = POSTaggerME.train(languageCode.map(_.toLower), sentenceStream, ModelType.MAXENT,
      null, null, eventCutoff, numIters);
  }
  
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def predict(testData: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
    val numWords  = wordIntMap.size
    val numTokens = testData.length
    var resultPair = new ArrayBuffer[Array[Boolean]](numTokens + 1)

    val tagger = new POSTaggerME(model);
    var sentenceStream = new TaggedSentenceStream(testData.iterator)

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
      sentenceStream = new TaggedSentenceStream(testData.iterator)
      val evaluator =
          new POSEvaluator(new opennlp.tools.postag.POSTaggerME(model));
      System.out.print("Evaluating ... ");
      evaluator.evaluate(sentenceStream);
      System.out.println("Accuracy: " + evaluator.getWordAccuracy());
    }

    resultPair
  }

}
