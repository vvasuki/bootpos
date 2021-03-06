package opennlp.bootpos.app
import opennlp.bootpos.util.collection._
import scala.collection.mutable.ArrayBuffer
// import ch.qos.logback._
import org.slf4j.LoggerFactory

object BootPos {
//  The file whence parameters such as laungage, corpus, taggerType are read.
  var RUNTIME_SETTINGS_FILE = getClass.getResource("/default/runtimeSettings.properties").getPath

//The following run-time settings are described in detail in the file RUNTIME_SETTINGS_FILE
  var bUniversalTags = false
  var bUseTrainingData = true; var bWiktionary = false;
  var bRawDataFromTrainingFile = false
  var testCorpus = ""
  var taggerType = ""
  var conllCorpora : List[String] = null
  var allCorpora : List[String] = null
  var bUseAsDictionary = false
  var DATA_DIR = ""
  val props = new java.util.Properties
  var numIterations = 1
  var rawTokensLimit = 0
  var taggedTokensLimit = 0
  var testTokensLimit = 0

  // Initial read of properties.
  // This is wverwritten if a fileName is passed during invocation.
  readRuntimeSettings

//  Read RUNTIME_SETTINGS_FILE and set parameters liek language, corpus etc..
  def readRuntimeSettings = {
    val file = new java.io.FileInputStream(RUNTIME_SETTINGS_FILE)
    props.load(file)
    file.close
    
    bUniversalTags = props.getProperty("bUniversalTags").toBoolean
    bUseTrainingData = props.getProperty("bUseTrainingData").toBoolean
    bUseAsDictionary =props.getProperty("bUseAsDictionary").toBoolean
    bWiktionary = props.getProperty("bWiktionary").toBoolean
    bRawDataFromTrainingFile = props.getProperty("bRawDataFromTrainingFile").toBoolean
    testCorpus = props.getProperty("testCorpus")
    taggerType = props.getProperty("taggerType")
    DATA_DIR = props.getProperty("DATA_DIR")
    numIterations = props.getProperty("numIterations").toInt
    conllCorpora = props.getProperty("conllCorpora").replace(" ", "").split(",").toList
    allCorpora = props.getProperty("allCorpora").replace(" ", "").split(",").toList
    rawTokensLimit = props.getProperty("rawTokensLimit").toInt
    taggedTokensLimit = props.getProperty("taggedTokensLimit").toInt
    testTokensLimit = props.getProperty("testTokensLimit").toInt
    

    if(bWiktionary) bUniversalTags = true;
    else bUseTrainingData = true;
  }

  def printSettings = {
    println("Properties file: "+ RUNTIME_SETTINGS_FILE)
    println("taggerType: "+ taggerType)
    println("Using universal tags? "+ bUniversalTags)
    println("Using wiktionary? "+ bWiktionary)
    println("bRawDataFromTrainingFile? "+ bRawDataFromTrainingFile)
    println("Using training data? "+ bUseTrainingData)
    println("Training tokens limit "+ taggedTokensLimit)
    println("Raw tokens limit "+ rawTokensLimit)

    if(bUseTrainingData)
      println("Using training data as dictionary ? "+ bUseAsDictionary)
  }
  
  /**
   * @param args the command line arguments:
   *  args(0), if it exists, is assumed to be RUNTIME_SETTINGS_FILE
   *
   */
  def main(args: Array[String]): Unit = {
    if(args.length>0) RUNTIME_SETTINGS_FILE = args(0)
    readRuntimeSettings
    printSettings
    def testOnCorpus(corpusStr : String) = {
      val Array(language, corpus) = corpusStr.split("-")
      println(language + " " + corpus)
      var wtp = new CorpusProcessor(language, corpus, taggerType)
      wtp.test
    }
    val results = new MatrixBufferDense[String](0, 0)
    testCorpus match {
    case "all" => {
        allCorpora.map(x => results addRow testOnCorpus(x))
      }
    case x => results addRow testOnCorpus(x)
    }
    println(results.toTsv)
    println(results.transpose.toTsv)
  }
}
object BootPosTest extends {
  val log = LoggerFactory.getLogger(this.getClass)
  def main(args: Array[String]): Unit = {
    log info( "Testing")
    collectionsTest.vpTest
    collectionsTest.serializabilityTest
    
  }
}