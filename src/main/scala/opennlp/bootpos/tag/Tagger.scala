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
class IntRepresentor {
  val log = LoggerFactory.getLogger(this.getClass)
  // We want to allow wordIntMap and tagIntMap to be supplied
  // while building compound taggers.
  var wordIntMap = new BijectiveHashMap[String, Int]
  var tagIntMap = new BijectiveHashMap[String, Int]

  // Initial guesses about number of tags and words
  val TAGNUM_IN = 25
  val WORDNUM_IN = 3000
  
  // The below quantity is used, together with wordIntMap,
  // to identify words not seen during the training phase.
  var numWordsTraining = 0
  var numWordsSeen = 0

  // The below is used to determine if wordIntMap should be blind to case.
  var bIgnoreCase = true

  var sentenceSepTag = 0
  var sentenceSepWord = 0
  var sentenceSepTagStr = ""
  var sentenceSepWordStr = ""

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

  def setSentenceSeparators(sentenceSepTagStrIn: String, sentenceSepWordStrIn: String) = {
    sentenceSepTagStr = sentenceSepTagStrIn
    sentenceSepWordStr = sentenceSepWordStrIn
    sentenceSepTag = getTagId(sentenceSepTagStr)
    sentenceSepWord = getWordId(sentenceSepWordStr)
    //added sentenceSepWord (supposedly during training), hence the following.
    numWordsTraining = numWordsTotal
  }

  def getTagStr(tag: Int) = tagIntMap.getKey(tag).get
  def getWordStr(word: Int) = wordIntMap.getKey(word).get

  def isNonTraining(word: String) = getWordId(word) >= numWordsTraining
  def isUnseen(word: String) = getWordId(word) >= numWordsSeen
}

trait Tagger extends Serializable{
  val log = LoggerFactory.getLogger(this.getClass)
  var bestTagsOverall = new LinkedList[Int]()

  // The below will be set during training
  var intMap = new IntRepresentor()

  def numTags = intMap.numTags
  def numWordsTotal = intMap.numWordsTotal


//        var wordId = testData(i)(0)
//        var tagId = testData(i)(1)

  def tag(tokensIn: ArrayBuffer[String]): IndexedSeq[String]
  def getTagDistributions(tokens: ArrayBuffer[Int]) = {log error "undefined"}
}

abstract class TaggerTrainer(sentenceSepTagStr :String, sentenceSepWordStr: String) extends Serializable {
  val log = LoggerFactory.getLogger(this.getClass)
  val tagger: Tagger
  var intMap = tagger.intMap
  intMap.setSentenceSeparators(sentenceSepTagStr, sentenceSepWordStr)
  
  def numTags = intMap.numTags
  
  def train(iter: Iterator[Array[String]]): Tagger
  def trainWithDictionary(dictionary: Dictionary) = train(dictionary.lstData.toIterator: Iterator[Array[String]])
  def processUntaggedData(lstTokens : ArrayBuffer[String]) = {log warn "Doing nothing!"; tagger}
}


