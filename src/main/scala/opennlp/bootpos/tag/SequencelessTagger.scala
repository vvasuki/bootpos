package opennlp.bootpos.tag

import scala.collection.mutable.HashMap
import scala.collection.immutable.Set
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import java.util.NoSuchElementException
import opennlp.bootpos.util.collection._

class SequencelessTagger extends Tagger {
  // The below can be optimized to have:
  //     O(1) lookup time for finding max_t Pr(t|w).

  //  Confidence in correctness: High.
//  Confidence in correctness: High.
//  Reason: Well tested.
  def tag(tokensIn: ArrayBuffer[String])= {
  //  Confidence in correctness: High.
  //  Reason: Well tested.
    def getBestTag(word: Int):Int = {
      if(intMap.isNonTraining(word)) bestTagsOverall.head
      else bestTagsByFreq(word).head
    }
    val testData = tokensIn.map(intMap.getWordId)
    testData.map(x => intMap.getTagStr(getBestTag(x)))
  }

}

class SequencelessTaggerTrainer(sentenceSepTagStr: String, sentenceSepWordStr: String) extends TaggerTrainer(sentenceSepTagStr, sentenceSepWordStr){
  override val tagger = new SequencelessTagger
  setIntMap
  
//  Confidence in correctness: High.
//  Reason: Well tested.
  def train(iter: Iterator[Array[String]]) = {
    intMap.updateWordTagList(iter.map(x => Array(intMap getWordId x(0), intMap getTagId(x(1)) )) toList)
    updateBestTagsOverall
    updateBestTagsByFreq
    tagger
  }

}
