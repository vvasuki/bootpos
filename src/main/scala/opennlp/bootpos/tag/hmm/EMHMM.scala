package opennlp.bootpos.tag.hmm

import opennlp.bootpos.tag._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedList
import opennlp.bootpos.util.collection._

class EMHMM(sentenceSepTagStr :String, sentenceSepWordStr: String) extends HMM(sentenceSepTagStr, sentenceSepWordStr){
  def processUntaggedData(text: List[String]) = {
    val numTokens = text.length
    
  }
}