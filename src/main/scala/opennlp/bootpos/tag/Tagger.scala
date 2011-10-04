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
  // We want to allow wordIntMap and tagIntMap to be supplied
  // while building compound taggers.
  var wordIntMap = new BijectiveHashMap[String, Int]
  var tagIntMap = new BijectiveHashMap[String, Int]

  // The below quantity is used, together with wordIntMap,
  // to identify words not seen during the training phase.
  var numWordsTraining = 0

  // The below is used to determine if wordIntMap should be blind to case.
  var bIgnoreCase = true


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
  def getWordId(wordIn: String): Int = {
    var word = wordIn
    if(bIgnoreCase)
      word = word.map(_.toUpper)
    if(!wordIntMap.contains(word))
      wordIntMap.put(word, wordIntMap.size)
    wordIntMap(word)
  }

//        var wordId = testData(i)(0)
//        var tagId = testData(i)(1)

  def processUntaggedData(lstTokens : ArrayBuffer[String]) = {}
  def train(iter: Iterator[Array[String]])
  def test(testData: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]]

  def trainWithDictionary(dictionary: Dictionary) = train(dictionary.lstData.toIterator: Iterator[Array[String]])

//  Confidence in correctness: High.
//  Reason: Well tested.
  def getResults(testData: ArrayBuffer[Array[Int]], bestTags: Array[Int]) = {
    val numTokens = testData.length
    var resultPairs = new ArrayBuffer[Array[Boolean]](numTokens)
    val tagCountTest = new ExpandingArray[Double](numTags)
    val tagErrorCount = new ExpandingArray[Double](numTags)
    resultPairs = resultPairs.padTo(numTokens, null)
    for(tokenNum <- 0 to numTokens-1) {
      var token = testData(tokenNum)(0)
      val bNovel = token >= numWordsTraining
      val tagActual = testData(tokenNum)(1)
      val bCorrect = bestTags(tokenNum) == tagActual
      tagCountTest.addAt(tagActual, 1)
      if(!bCorrect) tagErrorCount.addAt(tagActual, 1)
      resultPairs(tokenNum) = Array(bCorrect, bNovel)
    }
    val tagErrorRate = (tagErrorCount zip tagCountTest).map(x => x._1/x._2)
    val tagMaxError = tagErrorRate.indexOf(tagErrorRate.max)
    println("tagErrorRate " + tagErrorRate)
    println("tagMaxError " + getTagStr(tagMaxError))
    resultPairs
  }
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
      if(word >= wordTagFrequencies.numRows)
            return  bestTagsOverall.head
      var tf = wordTagFrequencies(word)
      return getBestTagsFromArray(tf).head
    }
    val testData = testDataIn.map(x => Array(getWordId(x(0)), getTagId(x(1))))
    var resultPair = new ArrayBuffer[Array[Boolean]](testData.length)
    testData.indices.foreach(i => {
        val wordId = testData(i)(0)
        val tagId = testData(i)(1)
        val bNovel = wordId >= wordTagFrequencies.numRows
        val bCorrect = getBestTag(testData(i)(0)) == tagId
        resultPair += Array(bCorrect, bNovel)
      })
    resultPair
  }

}
