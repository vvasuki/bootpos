package opennlp.bootpos.tag.labelPropagation

import opennlp.bootpos.tag._
import upenn.junto.app._
import upenn.junto.config._
import upenn.junto.graph._
import scala.collection.mutable._
import opennlp.bootpos.util.collection._
import opennlp.bootpos.app._
import scala.collection.JavaConversions._

class LabelPropagationDict extends LabelPropagationTaggerBase{
  val wordAfterWordMap = new MatrixBufferTrieRows[Int](intMap.WORDNUM_IN)

  //  Confidence in correctness: High.
  //  Reason: proved correct.
  def updateWordAfterWordMap(tokenIter: Iterator[Int]) = {
    var prevToken = intMap.sentenceSepWord
    for(token <- tokenIter) {
      wordAfterWordMap.increment(token, prevToken)
      prevToken = token
    }
    wordAfterWordMap(intMap.sentenceSepWord, intMap.sentenceSepWord) = 0
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
//
//  Claims:
//    All necessary (w, p) edges are added.
//      No spurious (w, p) edge is added.
//    All necessary (w, t) edges are added.
//      No spurious (w, t) edge is added.
//    Every (w, p) edge has the right weight.
//    Every (w, t) edge has the right weight.
  def makeEdges: ListBuffer[Edge] = {
    val edges = new ListBuffer[Edge]()
    val numWords = wordAfterWordMap.numRows

    for(word <- (0 to numWords -1)) {
//      Add (w, p) edges
      var numOcc = wordAfterWordMap(word).values.sum
      wordAfterWordMap(word).foreach(x => {
        val context = intMap.getWordStr(x._1)
        edges += new Edge(nodeNamer.w(word), nodeNamer.c(context), x._2/numOcc.toDouble)
      })

    }
    edges ++= makeWordTagEdges
  }


//  Input: expectedLabels: A list for the following:
//    Test-set words should be associated with the appropriate 'expected label'.
//  Output: Create an undirected graph with:
//    Nodes corresponding to a] W: all test and training word-types,
//      b] T: all tags with the corresponding tag-label, c] P: all word-types which preceed another word-type in the training data.
//
//    Edges a] (w, p) \in W \times P, with weight corresponding to
//      estimated probability of (p, w) sequences.
//      b] (w, t) with strength corresponding to estimated probability of w having a tag t.
//        In the case of novel words, simply use the uniform distribution on all possible tags excluding the sentence separator tag.
//        In case of known words, this would be derived from wordTagMap.
//
//    ExpectedLabels described earlier.
//
//
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def getGraph(expectedLabels: List[Label] = List()) : Graph = {
    val labels = getTagLabels
    val edges = makeEdges
/*    println("edges:")
    edges.foreach(println)
    println("labels:" + labels)
    println("expectedLabels:" + expectedLabels)*/
    val graph = GraphBuilder(edges.toList, labels.toList, expectedLabels)
    graph
  }

//  Get tag labels from graph.
//  Confidence in correctness: Moderate.
//  Reason: Proved correct but test on ic database fails to produce expected results.
  def getPredictions(graph: Graph) = {

    val wtMap = new HashMap[String, String]
    import scala.collection.JavaConverters._
    val nodeNames = graph.vertices.keySet
    nodeNames.filter(_.startsWith(nodeNamer.P_WORD_TYPE)).
      filterNot(_ == nodeNamer.w(intMap.sentenceSepWord)).
      foreach(x => {
      val v = graph.GetVertex(x)
      val tagStr = getBestLabel(v)
      wtMap(nodeNamer.deprefixify(x)) = tagStr
    })
    wtMap(intMap.sentenceSepWordStr) = intMap.sentenceSepTagStr
    wtMap
  }

//
//  Input: testData: ArrayBuffer[Array[Int]]: where testData(i) contains an array
//    where the first element is the word Id and the second element is the actual token.
//  Output: An ArrayBuffer of tuples (tag, bNovel) corresponding to each token in testData.
//
//  Confidence in correctness: Moderate.
//  Reason: Proved correct but test on ic database fails to produce expected results.
  def tag(testDataIn: ArrayBuffer[String]) = {
    val testData = testDataIn.map(intMap.getWordId)
    updateWordAfterWordMap(testData.iterator)
    var graph = getGraph()
    JuntoRunner(graph, 1.0, .01, .01, BootPos.numIterations, false)
    val wtMap = getPredictions(graph)

    var resultPair = new ArrayBuffer[Array[Boolean]](testData.length)
    testData.map(token => {
      val tokenStr = intMap.getWordStr(token)
      wtMap(tokenStr)
    })
  }

}

class LabelPropagationDictTrainer(sentenceSepTagStr :String, sentenceSepWordStr: String) extends TaggerTrainer(sentenceSepTagStr, sentenceSepWordStr) {
  override val tagger  = new LabelPropagationDict
  setIntMap
  val wordAfterWordMap = tagger.wordAfterWordMap

//  Input: word-token pairs from tagged text.
//  State alteration: Appropriately update the wordTagMap and wordAfterWordMap tables,
//    numTags and numWordsTraining.
//  Confidence in correctness: High.
//  Reason: proved correct.
  def train(iter: Iterator[Array[String]]) = {
    val lstData = iter.map(x => Array(intMap.getWordId(x(0)), intMap.getTagId(x(1)))).toList
    lstData.foreach(x => intMap.wordTagList.increment(x(0), x(1)))
    intMap.numWordsTraining = intMap.numWordsTotal
    updateBestTagsOverall
    tagger.updateWordAfterWordMap(lstData.map(_(0)).iterator)
    tagger
  }
}

