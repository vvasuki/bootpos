package opennlp.bootpos.tag.labelPropagation

import opennlp.bootpos.tag._
import opennlp.bootpos.util.io._
import opennlp.bootpos.util.collection._

class LabelPropagationTrainer(sentenceSepTagStr: String, sentenceSepWordStr: String) extends TaggerTrainer(sentenceSepTagStr, sentenceSepWordStr){
  override val tagger = new LabelPropagationTagger


  // Updates word-tag map, creates token edges.
  //  Confidence in correctness: High.
  //  Reason: proved correct.
  def train(iter: Iterator[Array[String]]) = {
    val lstData= iter.map(x => Array(intMap.getWordId(x(0)), intMap.getTagId(x(1)))).toList
    log info("training.. ")
    lstData.foreach(x => intMap.wordTagList.increment(x(0), x(1)))
    intMap.numWordsTraining = intMap.numWordsTotal
    updateBestTagsOverall
    // log info(txtIn.map(x => x(0) + " " + x(1)).mkString("\n"))
    tagger.addTokenEdges(lstData.map(_(0)), lstData.map(_(1)))
    tagger
  }


//  Input: word-token pairs from tagged text.
//  State alteration: Appropriately update the wordTagMap,
//    numWordsTraining.
//  Confidence in correctness: High.
//  Reason: proved correct.
  override def trainWithDictionary(dictionary: Dictionary) = {
    var lstData = dictionary.lstData.
      map(x => Array(intMap.getWordId(x(0)), intMap.getTagId(x(1))))
    lstData.foreach(x => intMap.wordTagList.increment(x(0), x(1)))
    intMap.numWordsTraining = intMap.numWordsTotal
    updateBestTagsOverall
    log info("numWordsTraining "+ intMap.numWordsTraining)
    // wordTagMap.matrix.foreach(x => log.info(x.indexWhere(y => y>0)))
    tagger
  }

}

