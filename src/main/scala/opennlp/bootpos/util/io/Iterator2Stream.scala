package opennlp.bootpos.util.io
import opennlp.tools.postag._
import opennlp.tools.util._
import scala.collection.mutable.ArrayBuffer
import scala.collection.immutable.Stream
import org.slf4j.LoggerFactory


//  Warning: The implementation of reset does not behave as expected.
//  This is because of the difficulty in constructing a stream from a scala iterator.
//  This is by design: we think that reset() will not be called.
class TaggedSentenceStream(wordTagIter: Iterator[Array[String]], sentenceSepTagStr: String) extends ObjectStream[POSSample] {
  val log = LoggerFactory.getLogger(this.getClass)

//  Confidence: Low.
//  Reason: Deliberately not well designed.
//  We assume that the defined funcitonality is not required.
  def close() = {log warn "Doing nothing!"}

//  Confidence: Low.
//  Reason: Deliberately not well designed.
//  We assume that the defined funcitonality is not required.
//    @throws(classOf[java.io.IOException])
  def reset() = {throw new java.io.IOException("Undefined operation.")}

//  Returns:
//    the next object or null to signal that the stream is exhausted.
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def read : POSSample = {
    if(wordTagIter.isEmpty) return null

    var words = new ArrayBuffer[String](20)
    var tags = new ArrayBuffer[String](20)

    wordTagIter.takeWhile(x => x(1) != sentenceSepTagStr).foreach(x => {
        words += x(0); tags += x(1)})
//      println(words)
//      println(tags)
    new POSSample(words.toArray, tags.toArray)
  }
}

