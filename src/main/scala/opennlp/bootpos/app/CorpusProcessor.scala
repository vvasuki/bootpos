package opennlp.bootpos.app

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util.io._
import opennlp.bootpos.tag._
import opennlp.bootpos.tag.hmm._
import opennlp.bootpos.tag.labelPropagation._
import java.util.NoSuchElementException
import java.io.File

class CorpusProcessor(language: String, corpus: String, taggerType: String = "WordTagProbabilities"){

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
  var tagger: Tagger = null
  taggerType match {
    case "OpenNLP" => {
      tagger = new OpenNLP( language, sentenceSepTag, sentenceSepWord)
    }
    case "HMM" => {
      tagger = new HMM(sentenceSepTag, sentenceSepWord)
    }
    case "EMHMM" => {
      tagger = new EMHMM(sentenceSepTag, sentenceSepWord, bUseTrainingStats = !(BootPos.bWiktionary || bTrainingDataAsDictionary))
      bProcessUntaggedData = true
    }
    case "LblPropEMHMM" => {
      tagger = new LblPropEMHMM(sentenceSepTag, sentenceSepWord, bUseTrainingStats = !(BootPos.bWiktionary || bTrainingDataAsDictionary))
      bProcessUntaggedData = true
    }
    case "LabelPropagation" => {
      tagger = new LabelPropagationTagger(sentenceSepTag, sentenceSepWord)
    }
    case _ => {
      tagger = new WordTagProbabilities(sentenceSepTag, sentenceSepWord)
    }
  }

  if(BootPos.bWiktionary) train(WIKTIONARY)
  if(BootPos.bUseTrainingData) train(TRAINING_DIR)



//  Confidence in correctness: High.
//  Reason: Well tested.
  def train(mode: String) = {
    println("Training with mode "+ mode)
    var iter = getWordTagIteratorFromFile(mode)
    val tokensUntagged = new ArrayBuffer[String]()

    if(bProcessUntaggedData){
      println("Processing untagged data.")
      val untaggedDataFile = getFileName("raw")
      tokensUntagged ++= new TextTableParser(file = untaggedDataFile, encodingIn = encoding, lineMapFn = lineMap(), maxLines = BootPos.rawTokensLimit).getColumn(0)
      if(tokensUntagged.head != sentenceSepWord) tokensUntagged prepend sentenceSepWord
      if(tokensUntagged.last != sentenceSepWord) tokensUntagged += sentenceSepWord
    }

    if(mode == WIKTIONARY) {
      // get words to consider.
      println("Loading test words too while picking dictionary entries.")
      val testWords = getWordTagIteratorFromFile(TEST_DIR).map(_(0)).toSet
      val dict = new Dictionary(iter, testWords ++ tokensUntagged)
      dict.addEntry(sentenceSepWord, sentenceSepTag)
      dict.updateCompleteness(tokensUntagged)
      tagger.trainWithDictionary(dict)
    }
    else if(!bTrainingDataAsDictionary) {
      if(BootPos.taggedTokensLimit > 0)
      tagger.train(iter.take(BootPos.taggedTokensLimit))
    }
    else {
      val dict = new Dictionary(iter)
      dict.removeDuplicateEntries
      dict.updateCompleteness(tokensUntagged)
      tagger.trainWithDictionary(dict)
    }
    if(bProcessUntaggedData){
      tagger.processUntaggedData(tokensUntagged)
    }

  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def test = {
    println("Testing " + language + ' ' + corpus);

    tagResults = new TaggingResult()
    val iter = getWordTagIteratorFromFile(TEST_DIR)
    val testData = new ArrayBuffer[Array[String]](10000)
    iter.copyToBuffer(testData)
    println("test tokens: " + testData.length)
    val results = tagger.test(testData)
    tagResults.processTaggingResults(results, testData, sentenceSepWord)

    tagResults.updateAccuracy
    // println("Most frequent tag overall: "+ tagger.bestTagsOverall)
    if(BootPos.bUniversalTags) println(tagMap.unmappedTags + " unmapped tags.")
    val corpusStr = language + "-" + corpus
    corpusStr + "\t"+tagResults.toTsv
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
    println("file: " + file)
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
    println("mode "+ mode)
    val file = getFileName(mode)

    var wordField = 1
    var tagField = 3;
    var sep = '\t'
    if(language.equals("danish")) tagField = 4
    if(mode.equals(WIKTIONARY))tagField = 2
    else if(! (BootPos.conllCorpora contains corpus)) {
      wordField = 0; tagField = 1;
      if(List("hmm") contains corpus)
        sep = '/'
    }


//      Prepare a function to map empty lines to an empty sentence word/ token pair.
    var newSentenceLine = sentenceSepWord;
    for(i <- 1 to tagField) newSentenceLine = newSentenceLine + sep + sentenceSepTag

//      Prepare a function to filter the lines from the stream based on whether they have the right number of fields and language-tags.
    var filterFn = ((x:Array[String]) => (x.length >= tagField+1))
    if(mode.equals(WIKTIONARY))
      filterFn = ((x:Array[String]) => ((x.length >= tagField+1) && x(0).equalsIgnoreCase(languageStr)))

    val parser = new TextTableParser(file = file, encodingIn = encoding, separator = sep, filterFnIn = filterFn, lineMapFn = lineMap(newSentenceLine))
    parser.getFieldIterator(wordField, tagField).map(x => {
        var tag = x(1); var word = x(0);
        if(BootPos.bUniversalTags) tag = tagMap.getMappedTag(tag, word)
        else tag = tag.map(_.toUpper)
        Array(word, tag)
      })
  }

}

