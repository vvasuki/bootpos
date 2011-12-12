package opennlp.bootpos.tag

import scala.collection.mutable.HashMap
import scala.collection.immutable.Set
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import scala.collection.IndexedSeq
import java.util.NoSuchElementException
import opennlp.bootpos.util.collection._
import org.slf4j.LoggerFactory

// The purpose of this class is to map all words and tags to integers,
// and to keep track of words seen during the training phase.
class IntRepresentor extends Serializable{
  val log = LoggerFactory.getLogger(this.getClass)
  // We want to allow wordIntMap and tagIntMap to be supplied
  // while building compound taggers.
  var wordIntMap = new BijectiveHashMap[String, Int]
  var tagIntMap = new BijectiveHashMap[String, Int]

  def numTags = tagIntMap.size
  def numWordsTotal = wordIntMap.size


  // Initial guesses about number of tags and words
  val TAGNUM_IN = 25
  val WORDNUM_IN = 3000
  
  // The below quantity is used, together with wordIntMap,
  // to identify words not seen during the training phase.
  var numWordsTraining = 0
  def updateNumWordsTraining = {
    numWordsTraining = numWordsTotal
  }
  var numWordsSeen = 0
  def updateNumWordsSeen = {
    numWordsTraining = numWordsSeen
  }
  def isNonTraining(wordId: Int): Boolean = wordId >= numWordsTraining
  def isNonTraining(word: String): Boolean = isNonTraining(getWordId(word))
  def isUnseen(word: String) = getWordId(word) >= numWordsSeen

  
  //  Confidence in correctness: High.
  //  Reason: Well tested.
  // The below is used to determine if wordIntMap should be blind to case.
  var bIgnoreCase = true
  def getWordId(wordIn: String): Int = {
    var word = wordIn
    if(bIgnoreCase)
      word = word.map(_.toUpper)
    if(!wordIntMap.contains(word))
      wordIntMap.put(word, wordIntMap.size)
    wordIntMap(word)
  }

  
  // Some frequently used values.
  var sentenceSepTag = 0
  var sentenceSepWord = 0
  var sentenceSepTagStr = ""
  var sentenceSepWordStr = ""
  def setSentenceSeparators(sentenceSepTagStrIn: String, sentenceSepWordStrIn: String) = {
    sentenceSepTagStr = sentenceSepTagStrIn
    sentenceSepWordStr = sentenceSepWordStrIn
    sentenceSepTag = getTagId(sentenceSepTagStr)
    sentenceSepWord = getWordId(sentenceSepWordStr)
    //added sentenceSepWord (supposedly during training), hence the following.
    updateNumWordsTraining
  }


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

  var wordTagList = new MatrixBufferListRows[Int](WORDNUM_IN, TAGNUM_IN)
//  Confidence in correctness: High.
//  Reason: Well tested.
  def possibleTags(token: Int) = {
    if(token< wordTagList.numRows)
      wordTagList(token).keys
    else (0 to numTags-1).filterNot(_ == sentenceSepTag)
  }
  def updateWordTagList(lstData: Seq[Array[Int]]) = {
    lstData.foreach(x => wordTagList.increment(x(0), x(1)))
    updateNumWordsTraining
    log info toString
  }

  override def toString = "numTags "+ numTags + " numWordsTraining " + numWordsTraining + " numWordsSeen " + numWordsSeen + " numWordsTotal " + numWordsTotal
}

trait Tagger extends Serializable{
  val log = LoggerFactory.getLogger(this.getClass)
  // A list of tags in decreasing order of frequency.
  var bestTagsOverall: Seq[Int] = null
  var bestTagsByFreq: ArrayBuffer[List[Int]] = null

  // The below will be set during training
  var intMap = new IntRepresentor()

  def numTags = intMap.numTags
  def numWordsTotal = intMap.numWordsTotal


//        var wordId = testData(i)(0)
//        var tagId = testData(i)(1)

  def tag(tokensIn: ArrayBuffer[String]): IndexedSeq[String]
  def getTagDistributions(tokens: ArrayBuffer[Int]) = {log error "undefined"; IndexedSeq[IndexedSeq[(Int, Double)]]()}
}

abstract class TaggerTrainer(sentenceSepTagStr :String, sentenceSepWordStr: String) extends Serializable {
  val log = LoggerFactory.getLogger(this.getClass)
  val tagger: Tagger = null
  var intMap: IntRepresentor = null
  def setIntMap = {
    intMap = tagger.intMap
    intMap.setSentenceSeparators(sentenceSepTagStr, sentenceSepWordStr)
  }
  
  def numTags = intMap.numTags
  
//  Confidence in correctness: High.
//  Reason: proved correct.
  def updateBestTagsOverall = {
    val tagCount = intMap.wordTagList.colSums
    tagger.bestTagsOverall = (0 to numTags-1) sortBy (t => -tagCount(t))
    log info (tagger.bestTagsOverall.map(intMap.getTagStr) mkString(" "))
  }

  def updateBestTagsByFreq ={tagger.bestTagsByFreq = intMap.wordTagList.matrix.map(
    lst => lst.filter(x => x._2 == lst.values.max).keys.toList
  )}

  def train(iter: Iterator[Array[String]]): Tagger
  def trainWithDictionary(dictionary: Dictionary) = train(dictionary.lstData.toIterator: Iterator[Array[String]])
  def processUntaggedData(lstTokens : ArrayBuffer[String]) = {log warn "Doing nothing!"; tagger}
}


