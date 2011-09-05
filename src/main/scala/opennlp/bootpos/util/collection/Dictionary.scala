package opennlp.bootpos.util.collection
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.HashSet

class Dictionary(iter: Iterator[Array[String]], wordsConsidered: Set[String] = null, sampleText:ArrayBuffer[String] = null){
/*    Filter away unnecessecary words for efficiency,
    and make a list of dictionary entries.*/
  val lstData = ListBuffer() ++ iter.filter(x =>
    if(wordsConsidered!= null)
      wordsConsidered.contains(x(0)) else true).toList

  val wordSet = new HashSet ++ lstData.map(_(0)).toSet

  // Gauge dictionary completeness.
  var completeness = 1.0
  if(sampleText != null) updateCompleteness(sampleText)

/*  Confidence in correctness: High
  Reason: proved correct.*/
  def updateCompleteness(tokens: ArrayBuffer[String]) = {
    var numTokens = tokens.length
    var numTokensSeen = 0
    tokens.foreach(x => {
      if(wordSet contains x) numTokensSeen += 1
    })
    completeness = numTokensSeen/ numTokens.toDouble
  }

/*  Confidence in correctness: High
  Reason: proved correct.*/
  def addEntry(word: String, descr: String) = {
    wordSet += word
    lstData += Array(word, descr)
  }

}