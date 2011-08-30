package opennlp.bootpos.tag

import scala.collection.mutable.HashMap
import scala.collection.immutable.Set
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import java.util.NoSuchElementException
import opennlp.bootpos.util.collection._

trait Tagger extends Serializable{
  val TAGNUM_IN = 25
  val WORDNUM_IN = 3000
  var bestTagsOverall = new LinkedList[Int]()
  val wordIntMap = new BijectiveHashMap[String, Int]
  val tagIntMap = new BijectiveHashMap[String, Int]


//  Confidence in correctness: High.
//  Reason: Well tested.
  def getTagId(tag: String): Int = {
    /*
     * Map a tag, update the mapping if necessary.
     */
    if(!tagIntMap.contains(tag))
      tagIntMap.put(tag, tagIntMap.size)
    tagIntMap(tag)
  }

  def getTagStr(tag: Int) = tagIntMap.getKey(tag).get
  def getWordStr(word: Int) = wordIntMap.getKey(word).get
  def numTags = tagIntMap.size
  def numWordsTotal = wordIntMap.size

//  Confidence in correctness: High.
//  Reason: Well tested.
  def getWordId(word: String): Int = {
    if(!wordIntMap.contains(word))
      wordIntMap.put(word, wordIntMap.size)
    wordIntMap(word)
  }

//        var wordId = testData(i)(0)
//        var tagId = testData(i)(1)

  def processUntaggedData(lstTokens : ArrayBuffer[String]) = {}
  def train(iter: Iterator[Array[String]])
  def test(testData: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]]

  def trainWithDictionary(iter: Iterator[Array[String]], wordSet: Set[String]) = train(iter: Iterator[Array[String]])
}

class WordTagProbabilities(sentenceSepTagStr :String, sentenceSepWordStr: String) extends Tagger {
  val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)

  val wordTagFrequencies = new MatrixBufferDense[Int](WORDNUM_IN, TAGNUM_IN)
  var tagFrequenciesOverall = new ExpandingArray[Int](TAGNUM_IN)

  //  Confidence in correctness: High.
  //  Reason: Well tested.
  def getBestTagsFromArray(tf: ExpandingArray[Int]):LinkedList[Int] = {
    new LinkedList[Int]() ++ tf.indices.filter((x) => (tf(x) == tf.max))
  }


//  Confidence in correctness: High.
//  Reason: Well tested.
  def train(iter: Iterator[Array[String]]) = {
    for(fields <- iter) {
      val word = getWordId(fields(0))
      val tag = getTagId(fields(1))
      tagFrequenciesOverall.addAt(tag, 1)
      wordTagFrequencies.increment(word, tag)
    }
    bestTagsOverall = getBestTagsFromArray(tagFrequenciesOverall)
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def test(testDataIn: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
  //  Confidence in correctness: High.
  //  Reason: Well tested.
    def getBestTag(word: Int):Int = {
      if(word >= wordTagFrequencies.length)
            return  bestTagsOverall.head
      var tf = wordTagFrequencies(word)
      return getBestTagsFromArray(tf).head
    }
    val testData = testDataIn.map(x => Array(getWordId(x(0)), getTagId(x(1))))
    var resultPair = new ArrayBuffer[Array[Boolean]](testData.length)
    testData.indices.foreach(i => {
        val wordId = testData(i)(0)
        val tagId = testData(i)(1)
        val bNovel = wordId >= wordTagFrequencies.length
        val bCorrect = getBestTag(testData(i)(0)) == tagId
        resultPair += Array(bCorrect, bNovel)
      })
    resultPair
  }

}
