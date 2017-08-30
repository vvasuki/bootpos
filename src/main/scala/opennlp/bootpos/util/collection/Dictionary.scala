package opennlp.bootpos.util.collection
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.HashSet
import org.slf4j.LoggerFactory

class Dictionary(iter: Iterator[Array[String]], wordsConsidered: Set[String] = null, sampleText:ArrayBuffer[String] = null){
  val log = LoggerFactory.getLogger(this.getClass)
/*    Filter away unnecessecary words for efficiency,
    and make a list of dictionary entries.*/
  var lstData = ListBuffer() ++ iter.filter(x =>
    wordsConsidered == null ||
    wordsConsidered.contains(x(0))).toList

  val wordSet = new HashSet ++ lstData.map(_(0)).toSet

  // Gauge dictionary completeness.
  var completeness = 1.0
  if(sampleText != null) updateCompleteness(sampleText)

/*  Confidence in correctness: High
  Reason: proved correct.*/
  def removeDuplicateEntries = {
    log info("Num entries (prior): "+ lstData.length)
    val delim = "~"
    val entrySet = lstData.map(x => x.mkString(delim)).toSet
    lstData = ListBuffer() ++ entrySet.map(x => x.split(delim)).toList
    log info("Num entries (after): "+ lstData.length)
  }

/*  Confidence in correctness: High
  Reason: proved correct.*/
  def updateCompleteness(tokens: ArrayBuffer[String]) = {
    var numTokens = tokens.length
    if(numTokens == 0) {
      log error "no tokens!"
      System.exit(1)
    }
    var numTokensSeen = 0
    tokens.foreach(x => {
      if(wordSet contains x) numTokensSeen += 1
    })
    completeness = numTokensSeen/ numTokens.toDouble
    log debug numTokensSeen + " in " + numTokens
    completeness = completeness * 0.9999
    log info("Dict completeness "+ completeness)
  }

/*  Confidence in correctness: High
  Reason: proved correct.*/
  def addEntry(word: String, descr: String) = {
    wordSet += word
    lstData += Array(word, descr)
  }

  override def toString = lstData.toString

}
