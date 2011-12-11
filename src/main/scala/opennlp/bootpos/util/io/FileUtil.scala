package opennlp.bootpos.util.io

import java.io.FileInputStream
import java.io._
import scala.io.Source
import opennlp.bootpos.app._
import opennlp.tools.sentdetect._
import opennlp.tools.tokenize._

object fileUtil {
/*  Confidence in correctness: High
  Reason: Well tested.*/
  def getFilePath(dir: String, condition: String => Boolean) = {
    // println("dir "+ dir)
    val filesInDir = new File(dir).list.toList
    // println("files " + filesInDir)
    val files = filesInDir.filter(condition).sorted
    dir + files.head
  }

  
/*  Confidence in correctness: High
  Reason: Adapted from internet.*/
  def write(fileName: String)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(new java.io.File(fileName))
    try { op(p) } finally { p.close() }
  }
}

object rawTextProcessor {
  val MODEL_PATH = BootPos.DATA_DIR + "/models/"

/*  Confidence in correctness: High
  Reason: Well tested.*/
  def loadTokenizer(language: String) = {
    val file = fileUtil.getFilePath(MODEL_PATH, (x => (x contains language) && (x contains "token")))
    val modelIn = new FileInputStream(file)
    val model = new TokenizerModel(modelIn);
    new TokenizerME(model)
  }

/*  Confidence in correctness: High
  Reason: Well tested.*/
  def loadSentenceDetector(language: String) = {
    val file = fileUtil.getFilePath(MODEL_PATH, (x => (x contains language) && (x contains "sent")))
    val modelIn = new FileInputStream(file)
    val model = new SentenceModel(modelIn);
    new SentenceDetectorME(model)
  }

/*  Confidence in correctness: High
  Reason: Well tested.*/
  def getSentences(language: String, rawTextFile: String) = {
    val sentenceDet = loadSentenceDetector(language)
    val source = Source.fromFile(rawTextFile)
    println("Got source!")
    val lines = source.getLines
    println("Got lines iterator!")
    lines.map(sentenceDet.sentDetect)
  }

/*  Confidence in correctness: High
  Reason: Well tested.*/
  def getTokens(language: String, rawTextFile: String) = {
    val tokenizer = loadTokenizer(language)
    val sentences = Source.fromFile(rawTextFile).getLines
    println("Got sentences iterator!")
    sentences.map(tokenizer.tokenize(_) ++ List(""))
  }
  
/*  Confidence in correctness: High
  Reason: Well tested.*/
  def main(args: Array[String]): Unit = {
    val Array(mode, language, rawTextFile, outFile) = args
    println(rawTextFile)
    val outStream = new PrintWriter(outFile)
    mode match {
      case "sent" => getSentences(language, rawTextFile).foreach(x => outStream.println(x.mkString("\n")))
      case "token" => getTokens(language, rawTextFile).foreach(x => outStream.println(x.mkString("\n")))
      case _ => println("Mode unrecognized.")
    }
  }
  
}
