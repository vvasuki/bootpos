package opennlp.bootpos.app
import opennlp.bootpos.util.collection._

object BootPos {
//  The file whence parameters such as laungage, corpus, taggerType are read.
  var RUNTIME_SETTINGS_FILE = getClass.getResource("/default/runtimeSettings.properties").getPath

//The following run-time settings are described in detail in the file RUNTIME_SETTINGS_FILE
  var bUniversalTags = false
  var bUseTrainingData = true; var bWiktionary = false;
  var language = ""; var corpus = ""; var taggerType = ""
  var DATA_DIR = ""

//  Read RUNTIME_SETTINGS_FILE and set parameters liek language, corpus etc..
  def readRuntimeSettings = {
    val file = new java.io.FileInputStream(RUNTIME_SETTINGS_FILE)
    val props = new java.util.Properties
    props.load(file)
    file.close
    bUniversalTags = props.getProperty("bUniversalTags").toBoolean
    bUseTrainingData = props.getProperty("bUseTrainingData").toBoolean
    bWiktionary = props.getProperty("bWiktionary").toBoolean
    language = props.getProperty("language")
    corpus = props.getProperty("corpus")
    taggerType = props.getProperty("taggerType")
    DATA_DIR = props.getProperty("DATA_DIR")

    if(bWiktionary) bUniversalTags = true;
    else bUseTrainingData = true;
    println("Properties file: "+ RUNTIME_SETTINGS_FILE)
    println(language + " " + corpus)
    println("Using universal tags? "+ bUniversalTags)
    println("Using wiktionary? "+ bWiktionary)
    println("Using training data? "+ bUseTrainingData)
  }
  
  /**
   * @param args the command line arguments:
   *  args(0), if it exists, is assumed to be
   *
   */
  def main(args: Array[String]): Unit = {
    if(args.length>0) RUNTIME_SETTINGS_FILE = args(0)
    readRuntimeSettings

//    collectionsTest.vpTest()
    var wtp = new CorpusProcessor(language, corpus, taggerType).test
  }

}
