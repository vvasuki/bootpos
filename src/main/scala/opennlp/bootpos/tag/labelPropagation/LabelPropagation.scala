package opennlp.bootpos.tag.labelPropagation

import opennlp.bootpos.tag._
import upenn.junto.app._
import upenn.junto.config._
import upenn.junto.graph._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ListMap
import scala.collection.immutable.IndexedSeq
import opennlp.bootpos.util.collection._
import opennlp.bootpos.app._

trait LabelPropagationTaggerBase extends Tagger{
  val nodeNamer = new NodeNamer(intMap)

  def tagsToPropagate = (0 to numTags-1) filterNot(_ == intMap.sentenceSepTag)

//    Set tag-node labels.
//    Exclude intMap.sentenceSepTag: we don't want it propagating.
//   Confidence: High
//   Reason: Well tested.
  def getTagLabels = {
    val labels = tagsToPropagate.map(x =>
      LabelCreator(nodeNamer.t(x), intMap.getTagStr(x))
    )
    labels
  }

//     Input: v: Vertex which is a word node, but is not intMap.sentenceSepWordStr.
  def getBestLabel(v: Vertex, possibleTags: IndexedSeq[String] = tagsToPropagate map intMap.getTagStr):String = {
    // log info("possibleTags " + possibleTags)
    val mostFrequentTag = intMap.getTagStr(bestTagsOverall.head)
    val scores = possibleTags map (v.GetEstimatedLabelScore(_))
    val maxScore = scores.max
    val minScore = scores.min

    // log info("Scores " + minScore + " " + maxScore)
    if(maxScore > minScore)
      possibleTags(scores.indices.find(scores(_) == maxScore).get)
    else{
      log error("getBestLabel: maxScore == minScore!")
      System.exit(1)
      mostFrequentTag
    }
  }

  //   Confidence: High
  //   Reason: Well tested.
  def makeWordTagEdges = {
    val edges = new ListBuffer[Edge]()
    val numWords = intMap.wordTagList.numRows
    // log info("intMap.wordTagList.numRows "+intMap.wordTagList.numRows)
    for(word <- 0 to numWords-1) {
//      Add (w, t) edges
//        In the case of novel words, simply use the uniform distribution on all possible tags excluding the sentence separator tag.
      if(word >= intMap.numWordsTraining)
        tagsToPropagate.foreach{
          x => edges += new Edge(nodeNamer.w(word), nodeNamer.t(x), 1/(numTags-1).toDouble)
        }
      else {
//        In case of known words, this would be derived from intMap.wordTagList.
//      Assumption : Every word w \in Training has atleast one tag associated with it.
        val numTaggings = intMap.wordTagList(word).values.sum
        // log info("possibleTags " + possibleTags)
        intMap.possibleTags(word).foreach(x =>
          edges += new Edge(nodeNamer.w(word), nodeNamer.t(x), intMap.wordTagList(word, x)/numTaggings.toDouble))
      }
    }
    // log info(edges.mkString("\n"))
    edges
  }

}

class LabelPropagationTagger extends LabelPropagationTaggerBase{

  // Updated using addTokenEdges
  val tokenEdges = new ListBuffer[Edge]()
  val tokenLabels = new ListBuffer[Label]()

  // Updated using addTokenEdges
  var numTokens = 0

//   Creates token-previousTokenType edges and token-wordType edges.
//  Confidence in correctness: High.
//  Reason: proved correct.
  def addTokenEdges(tokenList: List[Int], tagList: List[Int] = null) = {
    log info("adding token edges: ")
    var prevPrevToken = intMap.sentenceSepWord
    var prevToken = intMap.sentenceSepWord
    val tokenListLength = tokenList.length -1
    
    for(seqNum <- 0 to tokenListLength-1) {
      val token = tokenList(seqNum)
      if(seqNum % 100 == 0)
        printf("%1.2f\r", seqNum/tokenListLength.toDouble)
      // Prepare tokenId
      val tokenId = seqNum + numTokens

      //Create token-prevToken edge, update prevToken
      val context = prevPrevToken + "_" + prevToken
      tokenEdges += new Edge(nodeNamer.tok(tokenId), nodeNamer.c(context), 1)
      prevPrevToken = prevToken
      prevToken = token

      // Create token-tag or token-wordType edge.
      if(tagList != null)
        tokenLabels += LabelCreator(nodeNamer.tok(tokenId), nodeNamer.t(tagList(seqNum)))
        // tokenEdges += new Edge(nodeNamer.tok(tokenId), nodeNamer.t(tagList(seqNum)), 1)
      else
        tokenEdges += new Edge(nodeNamer.tok(tokenId), nodeNamer.w(prevToken), 1)
    }
    // update numTokens
    numTokens += tokenList.length
    log info("Done adding token edges.")
  }


// Make graph
// Confidence in correctness: High.
// Reason: Proved correct.
  def propagateLabels(tokens: ArrayBuffer[Int]) = {
    addTokenEdges(tokens.toList)
    val wordTagEdges = makeWordTagEdges
    // log debug(intMap.wordTagList)
    // log debug("wtEdges" + wordTagEdges.mkString("\n"))
    val edges = tokenEdges ++ wordTagEdges
    // log debug("tokenEdges " + tokenEdges.mkString("\n"))
    val labels = getTagLabels ++ tokenLabels
    // log debug("getTagLabels " + getTagLabels.mkString("\n"))
    // log debug("tokenLabels " + tokenLabels.mkString("\n"))
    val graph = GraphBuilder(edges.toList, labels.toList)
    // Run junto.
    JuntoRunner(graph, 1.0, .01, .01, BootPos.numIterations, false)
    graph
  }


//  Confidence in correctness: Hg.
//  Reason: .
  override def getTagDistributions(tokens: ArrayBuffer[Int]) = {
    log info("getPred ")
    val numPreTestTokens = numTokens
    val graph = propagateLabels(tokens)

    // Deduce tags.
    // Proved correct.
    val allTags = (0 to numTags - 1) map intMap.getTagStr
    val tagDistrForSentenceSeparators = IndexedSeq((intMap.sentenceSepTag, 1.0))
    
    val tagDistribution = tokens.indices.map(x => {
      val tokenId = numPreTestTokens + x
      val token = tokens(x)
      if(token == intMap.sentenceSepWord) {
        tagDistrForSentenceSeparators
      }
      else {
        val v = graph._vertices.get(nodeNamer.tok(tokenId))
        val scores = allTags.map(v.GetEstimatedLabelScore(_))
        val topTags = (allTags.indices zip scores).sortBy(x => x._2).takeRight(3)
        val sumScores = topTags.map(x => x._2).sum

        val distribution = topTags map (x => (x._1, x._2/sumScores))
        // log debug distribution.toString
        distribution
      }
    })
    tagDistribution
  }
  
// See the CHOICE-note below.
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def getPredictions(tokens: ArrayBuffer[Int]) = {
    log info("getPred ")
    val numPreTestTokens = numTokens
    val graph = propagateLabels(tokens)

    // Deduce tags.
    // Proved correct.
    val tagsFinal = tokens.indices.map(x => {
      val tokenId = numPreTestTokens + x
      val token = tokens(x)
      if(token == intMap.sentenceSepWord)
        intMap.sentenceSepTagStr
      else {
      // CHOICE: We are not checking the tag dictionary while
      // picking the label with the max score!
      // Perhaps this helps us overcome limitations in the dictionary.
        val v = graph._vertices.get(nodeNamer.tok(tokenId))
        getBestLabel(v)
      }
    })
    tagsFinal
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def tag(tokensIn: ArrayBuffer[String]) = {
    val testTokens = tokensIn.map(intMap.getWordId)
    getPredictions(testTokens)
  }

}

class NodeNamer(intMap: IntRepresentor) {
//  Prefixes distinguishing string names for various types of nodes.
//  Assumption: They are all of equal length.
  val P_WORD_TYPE = "W_"
  val P_CONTEXT = "C_"
  val P_TAG = "T_"
  val P_TOKEN = "V_"

//  For all three "convenience" functions:
//  Confidence in correctness: High.
//  Reason: proved correct.
  def w(id: Int) = P_WORD_TYPE + intMap.getWordStr(id)
  def c(context: String) = P_CONTEXT + context
  def t(id: Int) = P_TAG + intMap.getTagStr(id)
  def tok(id:Int) = P_TOKEN + id.toString

//  Confidence in correctness: High.
//  Reason: proved correct.
  def deprefixify(nodeName: String): String = nodeName.substring(P_TAG.length)
}

