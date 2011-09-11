package opennlp.bootpos.util.io
import scala.io.Source

class TextTableParser(file: String, encodingIn: String = "UTF-8", separator: Char = '\t', filterFnIn: Array[String] => Boolean = (x =>x.length >=1), lineMapFn : String => String = null, maxLines: Int = 0){
  
  val src = if(encodingIn == null) Source.fromFile(file)
    else {println("Encoding: " + encodingIn);
      Source.fromFile(file, encodingIn)}

  def getLines: Iterator[String]={
    var lines = src.getLines()
    if(maxLines>0) lines = lines.take(maxLines)
    if(lineMapFn != null) lines = lines.map(lineMapFn)
    lines
  }

//    Confidence in correctness: High
//    Reason: Used many times without problems.
  def getRowIterator: Iterator[Array[String]] = {
    val lines = getLines
    lines.map(_.split(separator)).filter(filterFnIn)
  }

//    Confidence in correctness: High
//    Reason: Used many times without problems.
  def getFieldIterator(fieldId1: Int, fieldId2: Int): Iterator[Array[String]] = {
    getRowIterator.map(x => Array(x(fieldId1), x(fieldId2)))
  }

  def getColumn(fieldId: Int = 0) = {
    getRowIterator.map(x => x(fieldId))
  }

//    Confidence in correctness: High
//    Reason: Proved correct.
  override protected def finalize : Unit = { src.close(); super.finalize}
}

