package opennlp.bootpos.app

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util.io._
import opennlp.bootpos.tag._
import opennlp.bootpos.tag.hmm._
import opennlp.bootpos.tag.labelPropagation._
import org.slf4j.LoggerFactory
import java.util.NoSuchElementException
import java.io.File

class CorpusProcessor(language: String, corpus: String, taggerType: String = "SequencelessTagger"){
  val log = LoggerFactory.getLogger(this.getClass)

  val DATA_DIR = BootPos.DATA_DIR
  val TEST_DIR = "test"
  val TRAINING_DIR = "train"
  val WIKTIONARY = "TEMP-S20110618.tsv"

  val sentenceSepTag = "###"
  val sentenceSepWord = "###"

  var tagResults = new TaggingResult()
  val tagMap = new TagMap(DATA_DIR+"universal_pos_tags.1.02/", language, corpus, sentenceSepTagIn = sentenceSepTag)
  var bProcessUntaggedData = false
  var bTrainingDataAsDictionary = BootPos.bUseAsDictionary

  var encoding = "UTF-8"
  language match {
    case "cz" => encoding = "ISO-8859-15"
    case _ => encoding = null
  }

  // Determine the full name of the language.
  val LANGUAGE_CODE_MAP = getClass.getResource("/lang/languageCodes.properties").getPath
  val props = new java.util.Properties
  val file = new java.io.FileInputStream(LANGUAGE_CODE_MAP)
  props.load(file)
  file.close
  val languageStr = props.getProperty(language, language)

//  Confidence in correctness: High.
//  Reason: Proved correct.
  var taggerTrainer: TaggerTrainer = null
  taggerType match {
    case "OpenNLP" => {
      taggerTrainer = new OpenNLPTrainer( language, sentenceSepTag, sentenceSepWord)
    }
    case "HMM" => {
      taggerTrainer = new HMMTrainer(sentenceSepTag, sentenceSepWord)
    }
    case "EMHMM" => {
      taggerTrainer = new EMHMM(sentenceSepTag, sentenceSepWord, bUseTrainingStats = !(BootPos.bWiktionary || bTrainingDataAsDictionary))
      bProcessUntaggedData = true
    }
    case "LblPropEMHMM" => {
      taggerTrainer = new LblPropEMHMM(sentenceSepTag, sentenceSepWord, bUseTrainingStats = !(BootPos.bWiktionary || bTrainingDataAsDictionary))
      bProcessUntaggedData = true
      if(!BootPos.bUseAsDictionary) {
        log error "Operation undefined"
        System.exit(1)
      }
    }
    case "LabelPropagation" => {
      taggerTrainer = new LabelPropagationTrainer(sentenceSepTag, sentenceSepWord)
    }
    case _ => {
      taggerTrainer = new SequencelessTaggerTrainer(sentenceSepTag, sentenceSepWord)
    }
  }

  // The following is set by the train method.
  var tagger: Tagger = null
  if(BootPos.bWiktionary) train(WIKTIONARY)
  if(BootPos.bUseTrainingData) train(TRAINING_DIR)

//  Confidence in correctness: High.
//  Reason: Well tested.
  def getUntaggedTokens = {
    val tokensUntagged = new ArrayBuffer[String]()
    log info("Processing untagged data.")
    if(BootPos.bRawDataFromTrainingFile) {
      var iterRaw = getWordTagIteratorFromFile("train")
      if(BootPos.rawTokensLimit > 0)
        iterRaw = iterRaw.take(BootPos.rawTokensLimit)
      tokensUntagged ++= iterRaw.map(_(0))
    }
    else {
      var untaggedDataFile = getFileName("raw")
      tokensUntagged ++= new TextTableParser(file = untaggedDataFile, encodingIn = encoding, lineMapFn = lineMap(), maxLines = BootPos.rawTokensLimit).getColumn(0)
    }
    log debug "Got " + tokensUntagged.length
    if(tokensUntagged.head != sentenceSepWord) tokensUntagged prepend sentenceSepWord
    if(tokensUntagged.last != sentenceSepWord) tokensUntagged += sentenceSepWord
    tokensUntagged
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def train(mode: String) = {
    log info("Training with mode "+ mode)
    var iterTraining = getWordTagIteratorFromFile(mode)
    var tokensUntagged = new ArrayBuffer[String]()

    if(bProcessUntaggedData || bTrainingDataAsDictionary){
      tokensUntagged = getUntaggedTokens
    }

    if(mode == WIKTIONARY) {
      // get words to consider.
      log info("Loading test words too while picking dictionary entries.")
      val testWords = getWordTagIteratorFromFile(TEST_DIR).map(_(0)).toSet
      val dict = new Dictionary(iterTraining, testWords ++ tokensUntagged)
      dict.addEntry(sentenceSepWord, sentenceSepTag)
      dict.updateCompleteness(tokensUntagged)
      tagger = taggerTrainer.trainWithDictionary(dict)
    }
    else {
      if(BootPos.taggedTokensLimit > 0)
        iterTraining = iterTraining.take(BootPos.taggedTokensLimit)
      if(!bTrainingDataAsDictionary) {
        tagger = taggerTrainer.train(iterTraining)
      }
      else {
        val dict = new Dictionary(iterTraining)
        dict.removeDuplicateEntries
        dict.updateCompleteness(tokensUntagged)
        log info "training with dictionary"
        tagger = taggerTrainer.trainWithDictionary(dict)
      }
    }
    if(bProcessUntaggedData){
      log info "processing Untagged Data"
      tagger = taggerTrainer.processUntaggedData(tokensUntagged)
    }
    tagger
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def test = {
    log info("Testing " + language + ' ' + corpus);

    var iter = getWordTagIteratorFromFile(TEST_DIR)
    if(BootPos.testTokensLimit > 0) iter = iter.take(BootPos.testTokensLimit)
    val testData = new ArrayBuffer[Array[String]](10000)
    iter.copyToBuffer(testData)
    log info("test tokens: " + testData.length)
    val tagResults = new TaggerTester(tagger).test(testData)
    // log info("Most frequent tag overall: "+ tagger.bestTagsOverall)
    if(BootPos.bUniversalTags) log info(tagMap.unmappedTags + " unmapped tags.")
    val corpusStr = language + "-" + corpus
    ArrayBuffer(corpusStr, ""+ tagResults.accuracy, ""+tagResults.accuracyKnown, ""+tagResults.accuracySeen, ""+tagResults.accuracyNovel, ""+tagResults.novelTokensFrac, tagResults.maxErrorTags)
  }



// ============== File processing below. Core logic above.




//  Confidence in correctness: High.
//  Reason: Well tested.
  def getFileName(fileType: String): String = {
    if(fileType == WIKTIONARY)
      return DATA_DIR + WIKTIONARY
    var languageCorpusString = language;
    languageCorpusString += '/' + corpus
    var dir = DATA_DIR + languageCorpusString

    val subDir = fileType.replace("raw", "train")
    dir += '/'+ subDir + '/'
    val file = fileUtil.getFilePath(dir, (x => (x contains fileType) && !(x contains ".ref")))
    log info("file: " + file)
    file
  }

/*  Purpose: While processing lines read from a file,
    replace empty lines with appropriate sentenceSeparator lines.
    If necessary, capitalize the line.*/
  def lineMap(newSentenceLine: String = sentenceSepWord)(x:String)= {
    var y = x.trim;
    if(y.isEmpty()) y= newSentenceLine;
    y
  }

//    @return Iterator[Array[String]] whose elements are arrays of size 2, whose
//      first element is the word and second element is the corresponding tag.
//    Confidence in correctness: High
//    Reason: Used many times without problems.
  def getWordTagIteratorFromFile(mode: String): Iterator[Array[String]] = {
//      Determine wordField, tagField, sep
    log info("mode "+ mode)
    val file = getFileName(mode)

    var wordField = 1
    var tagField = 3;
    var sep = '\t'
    if(language.equalsIgnoreCase("da")) tagField = 4
    if(mode.equals(WIKTIONARY))tagField = 2
    else if(! (BootPos.conllCorpora contains corpus)) {
      wordField = 0; tagField = 1;
      if(corpus endsWith "hmm")
        sep = '/'
    }


//      Prepare a function to map empty lines to an empty sentence word/ token pair.
    var newSentenceLine = sentenceSepWord;
    for(i <- 1 to tagField) newSentenceLine = newSentenceLine + sep + sentenceSepTag

//      Prepare a function to filter the lines from the stream based on whether they have the right number of fields and language-tags.
    var filterFn = ((x:Array[String]) => (x.length >= tagField+1))
    if(mode.equals(WIKTIONARY))
      filterFn = ((x:Array[String]) => ((x.length >= tagField+1) && x(0).equalsIgnoreCase(languageStr)))

    val parser = new TextTableParser(file = file, encodingIn = encoding,
      separator = sep, filterFnIn = filterFn, lineMapFn = lineMap(newSentenceLine))
    val iter = parser.getFieldIterator(wordField, tagField).map(x => {
        var tag = x(1); var word = x(0);
        if(BootPos.bUniversalTags) tag = tagMap.getMappedTag(tag, word)
        else tag = tag.map(_.toUpper)
        Array(word, tag)
      })
    log info tagMap.toString
    iter
  }

}

