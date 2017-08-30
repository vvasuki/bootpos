package opennlp.bootpos.util.collection

import opennlp.bootpos.util._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListMap
import scala.collection.Map
import scala.collection.immutable.HashMap
import java.util.NoSuchElementException
import scala.collection.generic._
import scala.collection.mutable.BufferLike
import scala.collection.mutable.IndexedSeqOptimized
import scala.collection.mutable.Builder
import scala.collection.mutable.ResizableArray
import org.slf4j.LoggerFactory

object matrixMath {
//  Confidence in correctness: High.
//  Reason: Tested multiple times
  def vp[T <: Seq[Double]](v1: T, v2: T): T = {
    v1.zip(v2).map(x => x._1 + x._2).asInstanceOf[T]
  }

//  Confidence in correctness: High.
//  Reason:Tested multiple times
  def vp[T <: Seq[Double]](v1: T, v2: Double): T = {
    v1.map(x => x + v2).asInstanceOf[T]
  }
}

/*
Extends ArrayBuffer, so that it has some special methods.
This buffer automatically grows to the required size when updated.
*/
class ExpandingArray[T](lengthIn: Int, defaultValue: T = null.asInstanceOf[T])  extends ArrayBuffer[T](lengthIn)
with GenericTraversableTemplate[T, ExpandingArray]
     with BufferLike[T, ExpandingArray[T]]
     with IndexedSeqOptimized[T, ExpandingArray[T]]
     with Builder[T, ExpandingArray[T]]
     with ResizableArray[T] {
//   The below few lines were copied from ArrayBuffer.scala in order to ensure that some common operations ( Eg: map?) return an ExpandingArray.
  override def companion: GenericCompanion[ExpandingArray] = ExpandingArray
  override def result: ExpandingArray[T] = this
  def this(v: Seq[T]) = {this(v.length);  ++=(v)}
  override def stringPrefix: String = "ExpArr"


//  Confidence in correctness: High.
//  Reason:Tested multiple times
  def addAt(idx: Int, diff: T)(implicit num: Numeric[T]) = {
    padTill(idx+1); update(idx, num.plus(apply(idx), diff))
  }

//  Confidence in correctness: High.
//  Reason:Tested multiple times
  def padTill(len: Int, value: T) =
    {(length to len-1).foreach((x) => +=(value))}
//  Confidence in correctness: High.
//  Reason:Tested multiple times
  def padTill(len: Int):Unit = padTill(len, defaultValue)

//  Confidence in correctness: High.
//  Reason:Tested multiple times
  override def update(index: Int, value: T) = {padTill(index + 1); super.update(index, value)}

}
object ExpandingArray extends SeqFactory[ExpandingArray] {
  implicit def canBuildFrom[A]: CanBuildFrom[Coll, A, ExpandingArray[A]] = new GenericCanBuildFrom[A]
  def newBuilder[A]: Builder[A, ExpandingArray[A]] = new ExpandingArray[A](16)
}

object collectionsTest{
  def vpTest(){
    var x = new ExpandingArray[Double](4)
    x ++= List(1, 2, 3, 4)
    var y = new ExpandingArray[Double](4)
    y ++= List(1.1, 2, 3, 4)
    y(9) = 9
    println(y)
    var z = matrixMath.vp(x,y)
    println(matrixMath.vp(z, 90))
  }

  def serializabilityTest = {
    var x = new ExpandingArray[Double](4)
    x ++= List(1, 2, 3, 4)
    var z1 = reflectionUtil.deepCopy(x)
    z1(2) = 2*z1(2)
    println(z1)
    var m = new MatrixBufferDense[Double](2, 2, 50)
    m(2, 2) = 100
    println(m)
    var m1 = reflectionUtil.deepCopy(m)
    m1(1, 1) = 2*m1(1,1)
    println(m1)
    
  }
}

/*
Stores a matrix, with some special behavior:
Its size is automatically increased to ensure update operations succeed.
*/
abstract class MatrixBuffer[T, X](rowsIn: Int, colsIn: Int, defaultValue: T = null.asInstanceOf[T], bSetInitSize: Boolean = false) extends Serializable{
  var matrix= new ExpandingArray[X](rowsIn)

//  State variable indicating maximum size of any row.
//  Confidence in correctness: High 
//  Reason: Proved correct.
  var numCols = 0
  if(bSetInitSize) updateSize(rowsIn, colsIn)

  def getEmptyRow: X

//  Confidence in correctness: High.
//  Reason: Proved correct, well tested.
  def padRows(numRowsIn: Int) =
    (matrix.length to numRowsIn-1).foreach((x) => matrix += (getEmptyRow))

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def numRows = matrix.length
  def size = Array(numRows, numCols)

  // Apply never fails due to there being too few columns.
  def apply(row: Int, col: Int): T

  def update(row: Int, col: Int, value: T)

// Methods to update matrix size using default values as necessary.
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def updateSize(nRows:Int, nCols: Int) = {numCols = List(nCols, numCols).max;
    padRows(nRows)}
  def expandBuffer(row: Int, col: Int) =updateSize(row+1, col+1)

  // def rowSums(implicit numeric: Numeric[T]) = matrix.map(_.sum)

  
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def getCol(col: Int) = (0 to numRows-1).map(x => apply(x, col))

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def colSum(col: Int)(implicit numeric: Numeric[T]) = {
    getCol(col).sum
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def colSums(implicit numeric: Numeric[T]) = {(0 to numCols-1).map(colSum(_))}

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def colFold(z: T)(col: Int, op: (T, T) => T)(implicit numeric: Numeric[T]) = {
    getCol(col).foldRight(z)(op)
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def addAt(row: Int, col: Int, diff: T)(implicit num: Numeric[T]) = {
    updateSize(row+1, col+1)
    update(row, col, num.plus(apply(row, col), diff))
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  override def toString(): String = matrix.toString.replace(matrix.stringPrefix, "\n")
}

class MatrixBufferDense[T] (rowsIn: Int, colsIn: Int, defaultValue: T = null.asInstanceOf[T], bSetInitSize: Boolean = false) extends MatrixBuffer[T, ExpandingArray[T]](rowsIn, colsIn, defaultValue, bSetInitSize){

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def setRow[Y<:  Seq[T]](row: Int, values : Y) = {
    matrix(row) = new ExpandingArray[T](values.length, defaultValue)
    matrix(row) ++= values
    updateSize(row+1, values.length)
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def addRow[Y <:  Seq[T]](values: Y ) = setRow(numRows, values)

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def setRows(M : IndexedSeq[IndexedSeq[T]]) = {
    M.indices.foreach(x => setRow(x, M(x)))
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def apply(row: Int, col: Int): T = {
    matrix(row).padTill(numCols, defaultValue)
    matrix(row)(col)
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def map [B] (f: (T) => B) = {
    val m = new MatrixBufferDense[B](numRows, numCols, f(defaultValue), bSetInitSize = true)
    m.matrix = matrix.map(_.map(f))
    m
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def apply(row: Int): ExpandingArray[T] = matrix(row)

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def getEmptyRow = new ExpandingArray[T](numCols, defaultValue)

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def padAllRows = matrix.foreach(_.padTill(numCols))

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def update(row: Int, col: Int, value: T) = {
    expandBuffer(row, col)
    matrix(row)(col) = value
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def increment(row: Int, col: Int)(implicit numeric: Numeric[T]) = {
    expandBuffer(row, col)
    // println(numeric.getClass)
    if(numeric.getClass.toString contains "Int"){
      // println("case Int")
      matrix(row)(col) = numeric.plus(apply(row, col), (1).asInstanceOf[T])
    }
    else{
      // println("case Double")
      matrix(row)(col) = numeric.plus(apply(row, col), (1.0).asInstanceOf[T])
    }
    
  }


//  Confidence in correctness: High.
//  Reason: Proved correct.
  def transpose = {
    val m = new MatrixBufferDense[T](numCols, numRows, defaultValue, bSetInitSize = true)
    (0 to numCols - 1) foreach (x => m.setRow(x, getCol(x)))
    m
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def toTsv = {
    matrix.map(_.mkString("\t")).mkString("\n")
  }
}


abstract class MatrixBufferMapRows[T](rowsIn: Int, colsIn: Int = 0, defaultValue: T = null.asInstanceOf[T], bSetInitSize: Boolean = false) extends MatrixBuffer[T, Map[Int, T]](rowsIn, colsIn, defaultValue, bSetInitSize) {
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def apply(row: Int, col: Int): T = if(col >= numCols) throw new NoSuchElementException(""+col)
    else return matrix(row).getOrElse(col, defaultValue)

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def apply(row: Int): Map[Int, T] = matrix(row)

}

// One way of having sparse rows.
// Each row is a immutable.HashMap where keys are stored in a trie/ prefix tree. So excessive memory is not wasted in storing sparse data.
// So each row is sparse.
class MatrixBufferTrieRows[T](rowsIn: Int, colsIn: Int = 0, defaultValue: T = null.asInstanceOf[T], bSetInitSize: Boolean = false) extends MatrixBufferMapRows(rowsIn, colsIn, defaultValue, bSetInitSize){

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def getEmptyRow = new HashMap[Int, T]()

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def update(row: Int, col: Int, value: T) = {
    expandBuffer(row, col)
    if(value == defaultValue) matrix(row) = matrix(row) - col
    else matrix(row) = matrix(row) + (col -> value)
  }
  
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def increment(row: Int, col: Int)(implicit numeric: Numeric[T]) = {
    expandBuffer(row, col)
    matrix(row) = matrix(row) + (col -> numeric.plus(apply(row, col), 1.asInstanceOf[T]))
  }
}

// Another way of having sparse rows.
class MatrixBufferListRows[T] (rowsIn: Int, colsIn: Int = 0, defaultValue: T = null.asInstanceOf[T], bSetInitSize: Boolean = false) extends MatrixBufferMapRows(rowsIn, colsIn, defaultValue, bSetInitSize){
//  Confidence in correctness: High.
//  Reason: Proved correct.
  def getEmptyRow = new ListMap[Int, T]()

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def update(row: Int, col: Int, value: T) = {
    expandBuffer(row, col)
    if(value == defaultValue) matrix(row) = matrix(row) - col
    else matrix(row) = matrix(row) + (col -> value)
  }

//  Confidence in correctness: High.
//  Reason: Proved correct.
  def increment(row: Int, col: Int)(implicit numeric: Numeric[T]) = {
    expandBuffer(row, col)
    matrix(row) = matrix(row) + (col -> numeric.plus(apply(row, col), 1.asInstanceOf[T]))
  }
}

