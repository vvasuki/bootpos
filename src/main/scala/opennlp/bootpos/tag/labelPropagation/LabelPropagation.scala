package opennlp.bootpos.tag.labelPropagation

import opennlp.bootpos.tag._
import upenn.junto.app._
import upenn.junto.config._
import upenn.junto.graph._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer
import scala.collection.immutable.IndexedSeq
import opennlp.bootpos.util.collection._
import opennlp.bootpos.app._

trait LabelPropagation extends Tagger{
  val sentenceSepTag = 0
  val wordTagMap = new MatrixBufferDense[Int](WORDNUM_IN, TAGNUM_IN)
  var numTrainingWords = 0

  val tagsToPropagate = (0 to numTags-1) filterNot(_ == sentenceSepTag)

//    Set tag-node labels.
//    Exclude sentenceSepTag: we don't want it propagating.
//   Confidence: High
//   Reason: Well tested.
  def getLabels = {
    var labels = tagsToPropagate map(x =>
      LabelCreator(nodeNamer.t(x), getTagStr(x))
    )
    labels
  }

//     Input: v: Vertex which is a word node, but is not sentenceSepWordStr.
  def getBestLabel(v: Vertex, possibleTags: IndexedSeq[String] = tagsToPropagate.map(getTagStr(_))):String = {
    val mostFrequentTag = getTagStr(bestTagsOverall.head)
    val scores = possibleTags map (v.GetEstimatedLabelScore(_))
    val maxScore = scores.max
    val minScore = scores.min

    // println(minScore + " " + maxScore)
    if(maxScore > minScore)
      possibleTags(scores.indices.find(scores(_) == maxScore).get)
    else{
      println("getBestLabel: maxScore == minScore!")
      mostFrequentTag
    }
  }

//   Confidence: High
//   Reason: Well tested.
  def updateWordTagMap(txtIn: Iterator[Array[Int]]) = {
    for(Array(token, tag) <- txtIn){
      wordTagMap.increment(token, tag)
    }
    numTrainingWords = wordTagMap.numRows
    updateBestTagsOverall
  }

//  Confidence in correctness: High.
//  Reason: proved correct.
  def updateBestTagsOverall = {
    val tagCount = wordTagMap.colSums
    bestTagsOverall = bestTagsOverall.+:(tagCount.indexOf(tagCount.max))
    // println(bestTagsOverall)
  }

//  Input: word-token pairs from tagged text.
//  State alteration: Appropriately update the wordTagMap,
//    numTrainingWords.
//  Confidence in correctness: High.
//  Reason: proved correct.
  override def trainWithDictionary(dictionary: Dictionary) = {
    var lstData = dictionary.lstData.
      map(x => Array(getWordId(x(0)), getTagId(x(1))))
    updateWordTagMap(lstData.iterator)
    // wordTagMap.matrix.foreach(x => println(x.indexWhere(_>0)))
  }

//   Confidence: High
//   Reason: Well tested.
  def makeWordTagEdges = {
    val edges = new ListBuffer[Edge]()
    val numWords = wordTagMap.numRows
    for(word <- 0 to numWords-1) {
//      Add (w, t) edges
//        In the case of novel words, simply use the uniform distribution on all possible tags excluding the sentence separator tag.
      if(word >= numTrainingWords)
        (0 to numTags-1).filter(x => x != sentenceSepTag).foreach{
          x => edges += new Edge(nodeNamer.w(word), nodeNamer.t(x), 1/(numTags-1).toDouble)
        }
      else {
//        In case of known words, this would be derived from wordTagMap.
//      Assumption : Every word w \in Training has atleast one tag associated with it.
        var numTaggings = wordTagMap(word).sum
        (0 to numTags-1).filter(wordTagMap(word, _) > 0).foreach(x =>
          edges += new Edge(nodeNamer.w(word), nodeNamer.t(x), wordTagMap(word, x)/numTaggings.toDouble))
      }
    }
    edges
  }

  object nodeNamer {
  //  Prefixes distinguishing string names for various types of nodes.
  //  Assumption: They are all of equal length.
    val P_WORD_TYPE = "W_"
    val P_PREVWORD_TYPE = "P_"
    val P_TAG = "T_"
    val P_TOKEN = "V_"

  //  For all three "convenience" functions:
  //  Confidence in correctness: High.
  //  Reason: proved correct.
    def w(id: Int) = P_WORD_TYPE + getWordStr(id)
    def p(id: Int) = P_PREVWORD_TYPE + getWordStr(id)
    def t(id: Int) = P_TAG + getTagStr(id)
    def tok(id:Int) = P_TOKEN + id.toString

  //  Confidence in correctness: High.
  //  Reason: proved correct.
    def deprefixify(nodeName: String): String = nodeName.substring(P_TAG.length)
  }

}

class LabelPropagationTagger(sentenceSepTagStr :String, sentenceSepWordStr: String) extends LabelPropagation{
  override val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)

  // Updated using addTokenEdges
  val tokenEdges = new ListBuffer[Edge]()

  // Updated using addTokenEdges
  var numTokens = 0

//   Creates token-previousTokenType edges and token-wordType edges.
//  Confidence in correctness: High.
//  Reason: proved correct.
  def addTokenEdges(tokenList: List[Int], tagList: List[Int] = null) = {
    var prevToken = sentenceSepWord
    for(seqNum <- tokenList.indices;
      token = tokenList(seqNum)) {
      // Prepare tokenId
      val tokenId = seqNum + numTokens

      //Create token-prevToken edge, update prevToken
      tokenEdges += new Edge(nodeNamer.tok(tokenId), nodeNamer.p(prevToken), 1)
      prevToken = token

      // Create token-tag or token-wordType edge.
      if(tagList != null)
        tokenEdges += new Edge(nodeNamer.tok(tokenId), nodeNamer.t(tagList(seqNum)), 1)
      else
        tokenEdges += new Edge(nodeNamer.tok(tokenId), nodeNamer.w(prevToken), 1)
    }
    // update numTokens
    numTokens += tokenList.length
  }

  // Updates word-tag map, creates token edges.
  //  Confidence in correctness: High.
  //  Reason: proved correct.
  def train(iter: Iterator[Array[String]]) = {
    val txtIn = iter.map(x => Array(getWordId(x(0)), getTagId(x(1)))).toList
    updateWordTagMap(txtIn.iterator)
    addTokenEdges(txtIn.map(_(0)), txtIn.map(_(1)))
  }

// See the CHOICE-note below.
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def getPredictions(tokensStr: ArrayBuffer[String]) = {
    val tokens = tokensStr.map(x => getWordId(x))

    // Make graph
    // Proved correct.
    addTokenEdges(tokens.toList)
    val edges = tokenEdges ++ makeWordTagEdges
    val labels = getLabels
    // TODO: Check the call and the result below.
    val graph = GraphBuilder(edges.toList, labels.toList, null)
    // Run junto.
    JuntoRunner(graph, 1.0, .01, .01, BootPos.numIterations, false)

    // Deduce tags.
    // Proved correct.
    val tagsFinal = tokens.indices.map(x => {
      val tokenId = numTrainingWords + x
      val token = tokens(x)
      if(token == sentenceSepWord)
        sentenceSepTagStr
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
  def test(testDataIn: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
    val testData = testDataIn.map(x => Array(getWordId(x(0)), getTagId(x(1))))
    val testTokensStr = testDataIn.map(x => x(0))
    val tagsFinal = getPredictions(testTokensStr).map(x => getTagId(x))
    getResults(testData, tagsFinal.toArray)
  }

}

