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
  val wordTagFrequencies = new MatrixBufferDense[Int](intMap.WORDNUM_IN, intMap.TAGNUM_IN)

  //  Confidence in correctness: High.
  //  Reason: Well tested.
  def getBestTagsFromArray(tf: ExpandingArray[Int]):LinkedList[Int] = {
    new LinkedList[Int]() ++ tf.indices.filter((x) => (tf(x) == tf.max))
  }


//  Confidence in correctness: High.
//  Reason: Well tested.
  def tag(tokensIn: ArrayBuffer[String])= {
  //  Confidence in correctness: High.
  //  Reason: Well tested.
    def getBestTag(word: Int):Int = {
      if(word >= wordTagFrequencies.numRows)
            return  bestTagsOverall.head
      var tf = wordTagFrequencies(word)
      val bestTags = getBestTagsFromArray(tf)
      return bestTags.head
    }
    val testData = tokensIn.map(intMap.getWordId)
    var resultPair = new ArrayBuffer[Array[Boolean]](testData.length)
    testData.map(x => intMap.getTagStr(getBestTag(x)))
  }

}

class SequencelessTaggerTrainer(sentenceSepTagStr: String, sentenceSepWordStr: String) extends TaggerTrainer(sentenceSepTagStr, sentenceSepWordStr){
  override val tagger = new SequencelessTagger
  var tagFrequenciesOverall = new ExpandingArray[Int](intMap.TAGNUM_IN)
  
//  Confidence in correctness: High.
//  Reason: Well tested.
  def train(iter: Iterator[Array[String]]) = {
    for(fields <- iter) {
      val word = intMap.getWordId(fields(0))
      val tag = intMap.getTagId(fields(1))
      tagFrequenciesOverall.addAt(tag, 1)
      tagger.wordTagFrequencies.increment(word, tag)
    }
    tagger.bestTagsOverall = tagger.getBestTagsFromArray(tagFrequenciesOverall)
    tagger
  }

}
