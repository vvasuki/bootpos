package opennlp.bootpos.tag.labelPropagation

import opennlp.bootpos.tag._
import upenn.junto.app._
import upenn.junto.config._
import upenn.junto.graph._
import scala.collection.mutable._
import opennlp.bootpos.util.collection._
import opennlp.bootpos.app._


class LabelPropagationDict(sentenceSepTagStr :String, sentenceSepWordStr: String) extends LabelPropagation{
  override val sentenceSepTag = getTagId(sentenceSepTagStr)
  val sentenceSepWord = getWordId(sentenceSepWordStr)

  val wordAfterWordMap = new MatrixBufferRowSparse[Int](WORDNUM_IN)

//  Confidence in correctness: High.
//  Reason: proved correct.
  def updateWordAfterWordMap(tokenIter: Iterator[Int]) = {
    var prevToken = sentenceSepWord
    for(token <- tokenIter) {
      wordAfterWordMap.increment(token, prevToken)
      prevToken = token
    }
    wordAfterWordMap(sentenceSepWord, sentenceSepWord) = 0
  }

//  Input: word-token pairs from tagged text.
//  State alteration: Appropriately update the wordTagMap and wordAfterWordMap tables,
//    numTags and numTrainingWords.
//  Confidence in correctness: High.
//  Reason: proved correct.
  def train(iter: Iterator[Array[String]]) = {
    val txtIn = iter.map(x => Array(getWordId(x(0)), getTagId(x(1)))).toList
    updateWordTagMap(txtIn.iterator)
    updateWordAfterWordMap(txtIn.map(_(0)).iterator)
    updateBestTagsOverall
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
        edges += new Edge(nodeNamer.w(word), nodeNamer.p(x._1), x._2/numOcc.toDouble)
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
    val labels = getLabels
    val edges = makeEdges
/*    println("edges:")
    edges.foreach(println)
    println("labels:" + labels)
    println("expectedLabels:" + expectedLabels)*/
    val graph = GraphBuilder(edges.toList, labels.toList, expectedLabels)
    graph
  }

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
  def prepareGraphData(testData: ArrayBuffer[Array[Int]]): List[Label] = {
    var wordTagMapTest = new MatrixBufferDense[Int]((1.5*numTrainingWords).toInt, numTags)
    val tokenList = testData.map(x => x(0))
    var testWordSet = tokenList.toSet
    updateWordAfterWordMap(tokenList.iterator)

    testData.foreach(x=> wordTagMapTest.increment(x(0), x(1)))

    var expectedLabels = testWordSet.map(x => {
      val tagFreq = wordTagMapTest(x)
      val tags = tagFreq.indices.filter(y => tagFreq(y)>0);
      tags.map(y => {
        val tagStr = getTagStr(y);
        val tagPr = tagFreq(y)/tagFreq.sum.toDouble;
        new Label(nodeNamer.w(x), tagStr, tagPr);
      }).toList
    }).toList.flatten
    return expectedLabels
  }

//  Get tag labels from graph.
//  Confidence in correctness: Moderate.
//  Reason: Proved correct but test on ic database fails to produce expected results.
  def getPredictions(graph: Graph) = {

    val wtMap = new HashMap[String, String]
    import scala.collection.JavaConverters._
    val nodeNames = graph._vertices.keySet.asScala
    nodeNames.filter(_.startsWith(nodeNamer.P_WORD_TYPE)).
      filterNot(_ == nodeNamer.w(sentenceSepWord)).
      foreach(x => {
      val v = graph._vertices.get(x)
      val tagStr = getBestLabel(v)
      wtMap(nodeNamer.deprefixify(x)) = tagStr
    })
    wtMap(sentenceSepWordStr) = sentenceSepTagStr
    wtMap
  }

//
//  Input: testData: ArrayBuffer[Array[Int]]: where testData(i) contains an array
//    where the first element is the word Id and the second element is the actual token.
//  Output: An ArrayBuffer of tuples (tag, bNovel) corresponding to each token in testData.
//
//  Confidence in correctness: Moderate.
//  Reason: Proved correct but test on ic database fails to produce expected results.
  def test(testDataIn: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
    val testData = testDataIn.map(x => Array(getWordId(x(0)), getTagId(x(1))))


    var expectedLabels = prepareGraphData(testData)
    var graph = getGraph(expectedLabels)
    JuntoRunner(graph, 1.0, .01, .01, BootPos.numIterations, false)
    val wtMap = getPredictions(graph)

    var resultPair = new ArrayBuffer[Array[Boolean]](testData.length)
    testData.indices.foreach(i => {
      val Array(token, actualTag) = testData(i)
      val tokenStr = getWordStr(token)

      val bNovelWord = (token >= numTrainingWords)
//       println("tokenStr: "+ getWordStr(token)+ " tag "+ tagStr + "act: "+getTagStr(actualTag))
      val tagStr = wtMap(tokenStr)
      val bCorrect = tagIntMap(tagStr) == actualTag
      resultPair += Array(bCorrect, bNovelWord)
    })

    //val resultPair = testData map { (t,tag) => ... }

    resultPair
  }

}

