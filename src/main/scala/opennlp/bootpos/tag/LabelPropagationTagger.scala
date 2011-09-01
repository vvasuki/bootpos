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
    numTrainingWords = wordTagMap.numRows
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
      LabelCreator(nodeNamer.t(x), getTagStr(x))
    )

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
      var numWords = wordAfterWordMap.numRows

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
/*    println("edges:")
    edges.foreach(println)
    println("labels:" + labels)
    println("expectedLabels:" + expectedLabels)*/
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
  def test(testDataIn: ArrayBuffer[Array[String]]): ArrayBuffer[Array[Boolean]] = {
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

    var expectedLabels = prepareGraphData
    var graph = getGraph(expectedLabels)
    JuntoRunner(graph, 1.0, .01, .01, 10, false)
    
//      Get tag lables from graph.
    var resultPair = new ArrayBuffer[Array[Boolean]](testData.length)
    def getBestLabel(v: Vertex):String = {
      val tags = tagIntMap.keys.toList
      val scores = tags.map(v.GetEstimatedLabelScore(_))
      val maxScore = scores.max
      tags(scores.indices.find(scores(_) == maxScore).get)
    }
    for(Array(token, actualTag) <- testData) {
      val bNovelWord = (token >= numTrainingWords)
      val tagStr = getBestLabel(graph._vertices.get(nodeNamer.w(token)))
//       println("tokenStr: "+ getWordStr(token)+ " tag "+ tagStr + "act: "+getTagStr(actualTag))
      val bCorrect = getTagId(tagStr) == actualTag
      resultPair += Array(bCorrect, bNovelWord)
    }

    //val resultPair = testData map { (t,tag) => ... }

    resultPair
  }

  object nodeNamer {
  //  Prefixes distinguishing string names for various types of nodes.
  //  Assumption: They are all of equal length.
    val P_WORD = "W_"
    val P_PREVWORD = "P_"
    val P_TAG = "T_"

  //  For all three "convenience" functions:
  //  Confidence in correctness: High.
  //  Reason: proved correct.
    def w(id: Int) = P_WORD + getWordStr(id)
    def p(id: Int) = P_PREVWORD + getWordStr(id)
    def t(id: Int) = P_TAG + getTagStr(id)

  //  Confidence in correctness: High.
  //  Reason: proved correct.
    def getId(nodeName: String): String = nodeName.substring(P_TAG.length)
  }
}

