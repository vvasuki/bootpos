package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import opennlp.bootpos.tag.labelPropagation._
import opennlp.bootpos.app._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util._
import upenn.junto.app._

class LblPropEMHMM(sentenceSepTagStr :String, sentenceSepWordStr: String, bUseTrainingStats: Boolean = true) extends 
EMHMM(sentenceSepTagStr, sentenceSepWordStr, bUseTrainingStats = false) {
  val lblPropTagger = new LabelPropagationTagger(sentenceSepTagStr, sentenceSepWordStr)

//  TODO: Not updating word-tag map here for limiting possible tags during EM iterations.
//  Confidence in correctness: Medium.
//  Reason: Seems to be fine.
  override def trainWithDictionary(dictionary: Dictionary) = {
    lblPropTagger.trainWithDictionary(dictionary)
    System.exit(1)
  }

//  Confidence in correctness: Low.
//  Reason: Implementation incomplete.
  override def processUntaggedData(textIn: ArrayBuffer[String]) = {
    val textInUp = textIn.map(_.map(_.toUpper))
    val tokenSeq1 = textInUp.map(x => lblPropTagger.getWordId(x))
//     val tags = lblPropTagger.getPredictions(textInUp)
    System.exit(1)
    
/*    Ensure that the following, which leads to errors, does not happen above:
    wtMap(x) == sentenceSepTagStr for x != sentenceSepWordStr.*/
    
//     train(textInUp.indices.map(x => Array(textInUp(x), tags(x))).iterator)

    super.processUntaggedData(textInUp)
  }

}