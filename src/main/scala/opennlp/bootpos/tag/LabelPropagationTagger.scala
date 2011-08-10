package opennlp.bootpos.tag

import upenn.junto.app._
import upenn.junto.config._
import upenn.junto.graph._
import scala.collection.mutable._
import opennlp.bootpos.util.collection._

class LabelPropagationTagger(sentenceSepTagStr :String, sentenceSepWordStr: String) extends Tagger{
  val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)

  val wordAfterWordMap = new MatrixBufferRowSparse[Int](WORDNUM_IN)
  val wordTagMap = new MatrixBufferDense[Int](WORDNUM_IN, TAGNUM_IN)
  var numTrainingWords = 0
  var numTags = 0

//  Input: word-token pairs - from a dictionary or tagged text.
//  State alteration: Appropriately update the wordTagMap and wordAfterWordMap tables,
//    numTags and numTrainingWords.
//  Confidence in correctness: High.
//  Reason: proved correct.
  def train(iter: Iterator[Array[String]]) = {
    var prevToken = sentenceSepWord
    for(Array(token, tag) <- iter.map(x => Array(getWordId(x(0)), getTagId(x(1))))){
      wordTagMap.increment(token, tag)
      wordAfterWordMap.increment(token, prevToken)
      prevToken = token
    }
    wordAfterWordMap(sentenceSepWord, sentenceSepWord) = 0
    numTrainingWords = wordTagMap.length
    numTags = wordTagMap.numCols
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
  def getGraph(expectedLabels: List[Label]) : Graph = {

//    Set tag-node labels.
    var labels = (0 to numTags-1) map(x => 
      LabelCreator(nodeNamer.t(x), x.toString))

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
      var edges = new ListBuffer[Edge]()
      var numWords = wordAfterWordMap.length

      for(word <- (0 to numWords -1)) {
  //      Add (w, p) edges
        var numOcc = wordAfterWordMap(word).values.sum
        wordAfterWordMap(word).foreach(x => {
          edges += new Edge(nodeNamer.w(word), nodeNamer.p(x._1), x._2/numOcc.toDouble)
        })

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

    val edges = makeEdges
    println("edges:")
    edges.foreach(println)
    println("labels:" + labels)
    println("expectedLabels:" + expectedLabels)
    val graph = GraphBuilder(edges.toList, labels.toList, expectedLabels)
    graph
  }


//
//  Input: testData: ArrayBuffer[Array[Int]]: where testData(i) contains an array
//    where the first element is the word Id and the second element is the actual token.
//  Output: An ArrayBuffer of tuples (tag, bNovel) corresponding to each token in testData.
//
//  Confidence in correctness: Low.
//  Reason: Proved correct but test on ic database fails to produce expected results.
  def predict(testDataIn: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
    val testData = testDataIn.map(x => Array(getWordId(x(0)), getTagId(x(1))))

//  Tasks:
//    Update wordAfterWordMap with information from testData.
//    Return the expected labels list.
//  Confidence in correctness: High.
//  Reason: Proved correct.
//  Claims:
//    wordAfterWordMap is updated using testData.
//    For each word occuring in testData, 
//      there is a Label in the expectedTags list.
//      This Label is correctly chosen based on actual tags
//      observed in the test data.
    def prepareGraphData: List[Label] = {
      var wordTagMapTest = new MatrixBufferDense[Int]((1.5*numTrainingWords).toInt, numTags)
      var testWordSet = new HashSet[Int]()

      var prevToken = sentenceSepWord
      for(Array(token, actualTag) <- testData) {
        wordAfterWordMap.increment(token, prevToken)

        wordTagMapTest.increment(token, actualTag)
        testWordSet += token

        prevToken = token
      }
      wordAfterWordMap(sentenceSepWord, sentenceSepWord) = 0

      var expectedTags = testWordSet.map(x => {
        var tags = wordTagMapTest(x);
        var bestTag = tags.indexOf(tags.max)
        LabelCreator(nodeNamer.w(x), bestTag.toString)}).toList
      return expectedTags
    }

    var expectedLabels = prepareGraphData
    var graph = getGraph(expectedLabels)
    JuntoRunner(graph, 1.0, .01, .01, 5, false)
    
    var resultPair = new ArrayBuffer[Array[Boolean]](testData.length)
//    resultPair = resultPair.padTo(testData.length, (0, false))
//      Get tag lables from graph.
    for(Array(token, actualTag) <- testData) {
      val bNovelWord = (token >= numTrainingWords)
      println(token + " ** " + graph._vertices.get(nodeNamer.w(1)))
      val tagStr = graph._vertices.get(nodeNamer.w(token)).getEstimatedLabelBest()
      println("--- "   + graph._vertices.get(nodeNamer.w(1)).GetEstimatedLabelScore("1"))
      val bCorrect = nodeNamer.getId(tagStr) == actualTag
      println("token: "+ token + " tag "+ tagStr)
      resultPair += Array(bCorrect, bNovelWord)
    }

    //val resultPair = for(Array(token, actualTag) <- testData) yield {
    //  val bNovelWord = (token >= numTrainingWords)
    //  val tagStr = graph._vertices.get(nodeNamer.w(token)).getEstimatedLabelBest()
    //  val bCorrect = nodeNamer.getId(tagStr) == actualTag
    //  println("token: "+ token + " tag "+ tagStr)
    //  (bCorrect, bNovelWord)
    //}

    //val resultPair = testData map { (t,tag) => ... }

    resultPair
  }

}

object nodeNamer {
//  Prefixes distinguishing string names for various types of nodes.
//  Assumption: They are all of equal length.
  val P_WORD = "W_"
  val P_PREVWORD = "P_"
  val P_TAG = "T_"
  
//  Confidence in correctness: High.
//  Reason: proved correct.
  def getStringName(id: Int, nodeType: String) = nodeType + id

//  For all three "convenience" functions:
//  Confidence in correctness: High.
//  Reason: proved correct.
  def w(id: Int) = getStringName(id, P_WORD)
  def p(id: Int) = getStringName(id, P_PREVWORD)
  def t(id: Int) = getStringName(id, P_TAG)

//  Confidence in correctness: High.
//  Reason: proved correct.
  def getId(nodeName: String): Int = nodeName.substring(P_TAG.length).toInt
}
