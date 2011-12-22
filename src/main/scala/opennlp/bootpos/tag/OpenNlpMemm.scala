package opennlp.bootpos.tag

import opennlp.bootpos.util.io._
import opennlp.tools.postag._
import opennlp.tools.util._
import opennlp.tools.util.model._
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.immutable.Stream

class OpenNLP extends Tagger{

  var model: POSModel = null
  
//  Confidence in correctness: Medium.
//  Reason: Unchecked.
  def tag(testData: ArrayBuffer[String]) = {
    val numTokens = testData.length

    val tagger = new POSTaggerME(model);
  //    update wordIntMap appropriately.
    testData.foreach(x => intMap.getWordId(x.map(_.toUpper))) // update wordIntMap.

    val testDataIter = testData.iterator
    def getSentence = testDataIter.takeWhile(x => x != intMap.sentenceSepWordStr).toArray
    val tags = new ArrayBuffer[String](testData.length)

    var sample = getSentence
    while(!sample.isEmpty){
      tags ++= tagger.tag(sample)
      // Add a result corresponding to correct tagging of sentence-separator word.
      if(1 <= testData.length - tags.length )
        tags += intMap.sentenceSepTagStr
      sample = getSentence
    }
    tags
  }

// openNLP's Evaluator
// Confidence: High
// Reason: Proved correct.
// Observed to yield same result as above.
//     def openNLPEval = {
//       sentenceStream = new TaggedSentenceStream(testData.iterator, sentenceSepTagStr)
//       val evaluator =
//           new POSEvaluator(new opennlp.tools.postag.POSTaggerME(model));
//       System.out.print("Evaluating ... ");
//       evaluator.evaluate(sentenceStream);
//       System.out.println("Accuracy: " + evaluator.getWordAccuracy());
//     }

}

class OpenNLPTrainer(languageCode: String, sentenceSepTagStr :String, sentenceSepWordStr: String) extends TaggerTrainer(sentenceSepTagStr, sentenceSepWordStr)
 {
  override val tagger = new OpenNLP
  setIntMap
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
    wordTagLst.foreach(x => intMap.getWordId(x(0).map(_.toUpper))) // update wordIntMap.
    intMap.updateNumWordsTraining

    val sentenceStream = new TaggedSentenceStream(wordTagLst.iterator, intMap.sentenceSepTagStr)
    val numIters = 100; val eventCutoff = 5
    tagger.model = POSTaggerME.train(languageCode.map(_.toLower), sentenceStream, ModelType.MAXENT,
      null, null, eventCutoff, numIters)
    tagger
  }
}
