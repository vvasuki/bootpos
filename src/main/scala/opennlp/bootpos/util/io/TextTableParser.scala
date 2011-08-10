package opennlp.bootpos.util.io
import scala.io.Source

class TextTableParser(file: String, separator: Char = '\t', filterFnIn: Array[String] => Boolean = (x =>x.length >=1), lineMapFn : String => String = null){
  val src = Source.fromFile(file)

//    Confidence in correctness: High
//    Reason: Used many times without problems.
  def getRowIterator: Iterator[Array[String]] = {
    var lines = src.getLines()
    if(lineMapFn != null) lines = lines.map(lineMapFn)
    lines.map(_.split(separator)).filter(filterFnIn)
  }

//    Confidence in correctness: High
//    Reason: Used many times without problems.
  def getFieldIterator(fieldId1: Int, fieldId2: Int): Iterator[Array[String]] = {
    getRowIterator.map(x => Array(x(fieldId1), x(fieldId2)))
  }

//    Confidence in correctness: High
//    Reason: Proved correct.
  override protected def finalize : Unit = { src.close(); super.finalize}
}

