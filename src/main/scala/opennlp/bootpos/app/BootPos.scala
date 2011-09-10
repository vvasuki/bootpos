package opennlp.bootpos.app
import opennlp.bootpos.util.collection._
import scala.collection.mutable.ArrayBuffer

object BootPos {
//  The file whence parameters such as laungage, corpus, taggerType are read.
  var RUNTIME_SETTINGS_FILE = getClass.getResource("/default/runtimeSettings.properties").getPath

//The following run-time settings are described in detail in the file RUNTIME_SETTINGS_FILE
  var bUniversalTags = false
  var bUseTrainingData = true; var bWiktionary = false;
  var testCorpus = ""
  var taggerType = ""
  var conllCorpora : List[String] = null
  var allCorpora : List[String] = null
  var DATA_DIR = ""
  val props = new java.util.Properties
  var numIterations = 1
  var rawTokensLimit = 0

//  Read RUNTIME_SETTINGS_FILE and set parameters liek language, corpus etc..
  def readRuntimeSettings = {
    val file = new java.io.FileInputStream(RUNTIME_SETTINGS_FILE)
    props.load(file)
    file.close
    bUniversalTags = props.getProperty("bUniversalTags").toBoolean
    bUseTrainingData = props.getProperty("bUseTrainingData").toBoolean
    bWiktionary = props.getProperty("bWiktionary").toBoolean
    testCorpus = props.getProperty("testCorpus")
    taggerType = props.getProperty("taggerType")
    DATA_DIR = props.getProperty("DATA_DIR")
    numIterations = props.getProperty("numIterations").toInt
    conllCorpora = props.getProperty("conllCorpora").replace(" ", "").split(",").toList
    allCorpora = props.getProperty("allCorpora").replace(" ", "").split(",").toList
    rawTokensLimit = props.getProperty("rawTokensLimit").toInt

    if(bWiktionary) bUniversalTags = true;
    else bUseTrainingData = true;
    println("Properties file: "+ RUNTIME_SETTINGS_FILE)
    println("Using universal tags? "+ bUniversalTags)
    println("Using wiktionary? "+ bWiktionary)
    println("Using training data? "+ bUseTrainingData)
  }
  
  /**
   * @param args the command line arguments:
   *  args(0), if it exists, is assumed to be RUNTIME_SETTINGS_FILE
   *
   */
  def main(args: Array[String]): Unit = {
    if(args.length>0) RUNTIME_SETTINGS_FILE = args(0)
    readRuntimeSettings
    def testOnCorpus(corpusStr : String) = {
      val Array(language, corpus) = corpusStr.split("-")
      println(language + " " + corpus)
      var wtp = new CorpusProcessor(language, corpus, taggerType)
      wtp.test
    }
    val results = new ArrayBuffer[String](10)
    testCorpus match {
    case "all" => {
        results ++= allCorpora.map(testOnCorpus)
      }
    case x => results += testOnCorpus(x)
    }
    println(results.mkString("\n"))
  }
}
object BootPosTest {
  def main(args: Array[String]): Unit = {
    collectionsTest.vpTest
    collectionsTest.serializabilityTest
  }
}