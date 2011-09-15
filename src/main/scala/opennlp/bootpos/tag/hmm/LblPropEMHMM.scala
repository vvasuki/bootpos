package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import opennlp.bootpos.app._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._

class LblPropEMHMM(sentenceSepTagStr :String, sentenceSepWordStr: String, bUseTrainingStats: Boolean = true) extends 
EMHMM(sentenceSepTagStr, sentenceSepWordStr, bUseTrainingStats = false) {
  val lblPropTagger = new LabelPropagationTagger(sentenceSepTagStr, sentenceSepWordStr)

//  Confidence in correctness: Medium.
//  Reason: Seems to be fine.
  override def trainWithDictionary(dictionary: Dictionary) = {
    lblPropTagger.trainWithDictionary(dictionary)
    this.tagIntMap = lblPropTagger.tagIntMap
    this.wordIntMap = lblPropTagger.wordIntMap
  }

  override def processUntaggedData(textIn: ArrayBuffer[String]) = {
    val textInUp = textIn.map(_.map(_.toUpper))
    val text = textInUp.map(x => getWordId(x))
    lblPropTagger.updateWordAfterWordMap(text.iterator)
    val graph = lblPropTagger.getGraph()
    val wtMap = lblPropTagger.getPredictions(graph)
    train(textInUp.map(x => Array(x, wtMap(x))).iterator)
    processTokenSeq(text)
  }

}