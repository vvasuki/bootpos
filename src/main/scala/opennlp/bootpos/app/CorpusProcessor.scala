package opennlp.bootpos.app

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util.io.TextTableParser
import opennlp.bootpos.tag._
import opennlp.bootpos.tag.hmm._
import java.util.NoSuchElementException
import java.io.File

class TaggingResult {
  var numTestTokensKnown = 0
  var numTestTokensSeen = 0
  var numTestTokensNovel = 0
  var correctTaggingsKnown = 0
  var correctTaggingsSeen = 0
  var correctTaggingsNovel = 0

//  Confidence in correctness: High.
//  Reason: Well tested.
  def update(bCorrect: Boolean, bNovelToken: Boolean, bSeenToken: Boolean) = {
    if(!bNovelToken) {
      if(bCorrect) correctTaggingsKnown  = correctTaggingsKnown  + 1
      numTestTokensKnown = numTestTokensKnown + 1
    }
    else if(bSeenToken){
      if(bCorrect) correctTaggingsSeen  = correctTaggingsSeen  + 1
      numTestTokensSeen = numTestTokensSeen + 1
    }
    else {
      if(bCorrect) correctTaggingsNovel  = correctTaggingsNovel  + 1
      numTestTokensNovel = numTestTokensNovel + 1
    }
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def getAccuracy() = {
    var correctTaggings = correctTaggingsKnown + correctTaggingsNovel
    var numTestTokens = numTestTokensKnown + numTestTokensNovel
    var accuracy = correctTaggings/ numTestTokens.toDouble
    var accuracyKnown = correctTaggingsKnown/ numTestTokensKnown.toDouble
    var accuracyNovel = correctTaggingsNovel/ numTestTokensNovel.toDouble

    printf("Accuracy: %.3f, (Known: %.3f, Novel: %.3f)\n", accuracy, accuracyKnown, accuracyNovel)
    printf("Non training tokens: %d, %.3f\n", numTestTokensNovel, numTestTokensNovel/numTestTokens.toDouble)
  }
  
//  Confidence in correctness: High.
//  Reason: Well tested.
  def processTaggingResults(results: ArrayBuffer[Array[Boolean]], testData: ArrayBuffer[Array[String]], sentenceSepWord: String)= {
    val bUntaggedTextUsed = results(0).length>2
    for {i <- testData.indices.iterator
      if(testData(i)(0) != sentenceSepWord)
    }{
      val tag = testData(i)(0);
      val bCorrect  = results(i)(0)

      // The token has been seen in tagged text.
      val bNovelToken = results(i)(1)

      // The token has been seen in untagged text - so not entirely novel.
      val bSeenToken = if(bUntaggedTextUsed) results(i)(2) else false
      update(bCorrect, bNovelToken, bSeenToken)
    }
  }
}

class TagMap(TAG_MAP_DIR: String, languageCode: String, corpus: String, sentenceSepTagIn: String = null) {
  val tagMap = new HashMap[String, String]()
  if(sentenceSepTagIn != null)
    tagMap(sentenceSepTagIn) = sentenceSepTagIn

  var unmappedTags = 0

  //X - other: foreign words, typos, abbreviations
  //ADP - adpositions (prepositions and postpositions)
  //. - punctuation
  val tagsUniversal = List("VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", ".")

//  Confidence in correctness: High.
//  Reason: Well tested.
  if(BootPos.bUniversalTags)  try {
    // Populate the tagMap by reading the appropriate file.
    var tagMapFile = TAG_MAP_DIR + languageCode + "-" + corpus + ".map"
    
    val parser = new TextTableParser(file = tagMapFile, filterFnIn = (x =>x.length >= 2), lineMapFn = (x => x.map(_.toUpper)))
    parser.getRowIterator.foreach(x => tagMap(x(0)) = x(1))
    tagMap.values.foreach(x => tagMap(x) = x)

  //  Add the universal tags themselves to the map.
    tagsUniversal.foreach(x => tagMap(x) = x)

//    print(tagMap)
  } catch {
    case e: java.io.FileNotFoundException => println("Alert: no tag map found!" + e)
  }

  /*
  * Add mapping for a tag to the tag map by the following procedure:
    1. Programmatically comparing with the universal tag set tags.
    2. If 1 fails, set tagMap(tag) = tag.
  * Assumption: There is no pre-existing mapping.
  * Arguments: tagIn: the tag to be added.
  *  word: A word corresponding to the tag; used for printing a helpful message.
  */
  //  Confidence in correctness: High.
  //  Reason: Well tested.
  def updateTagMap(tagIn: String, word: String): Unit= {
    var tag = tagIn.map(_.toUpper)
    if(tagMap.contains(tag)) throw new IllegalArgumentException("Mapping already exists")

    var iter = tagsUniversal.filter(x => !(x.equals("X") || x.equals(".")))
    for(tagUniversal <- iter) {
      if(tag.indexOf(tagUniversal) != -1) {tagMap(tag) = tagUniversal; return}
    }
    //    Begin special cases.
    // Mapping to universal tag sets.
    if(tag.indexOf("SYMBOL") != -1) {tagMap(tag) = "NOUN"; return}
    //ADP - adpositions (prepositions and postpositions)
    if(tag.indexOf("POSITION") != -1) {tagMap(tag) = "ADP"; return}
    //X - other: foreign words, typos, abbreviations
    if(tag.indexOf("ABBREV") != -1 || tag.indexOf("FOREIGN") != -1 || tag.indexOf("ACRONYM") != -1 || tag.indexOf("INITIAL") != -1) {tagMap(tag) = "X"; return}
    //. - punctuation
    if(tag.indexOf("PUNCTUAT") != -1) {tagMap(tag) = "X"; return}
    if(tag.indexOf("PARTICLE") != -1) {tagMap(tag) = "PRT"; return}
    if(tag.indexOf("ARTICLE") != -1) {tagMap(tag) = "DET"; return}
    unmappedTags = unmappedTags+1
    println("unknown tag "+tag + " :word "+ word);
    tagMap(tag) = "X";
  }
//  Confidence in correctness: High.
//  Reason: Proved.
  def getMappedTag(tagIn1: String, word: String): String = {
    var tag = tagIn1.map(_.toUpper)
    try{tag = tagMap(tag);}
    catch{
      case e => updateTagMap(tag, word)
    }
    return tag
  }

}

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
  var bIgnoreCase = true
  val bTrainingDataAsDictionary = true

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
      bIgnoreCase = false
    }
    case "HMM" => tagger = new HMM(sentenceSepTag, sentenceSepWord)
    case "EMHMM" => {
      tagger = new EMHMM(sentenceSepTag, sentenceSepWord, bUseTrainingStats = !(BootPos.bWiktionary || bTrainingDataAsDictionary))
      bProcessUntaggedData = true
      bIgnoreCase = true
    }
    case "LabelPropagation" => tagger = new LabelPropagationTagger(sentenceSepTag, sentenceSepWord)
    case _ => tagger = new WordTagProbabilities(sentenceSepTag, sentenceSepWord)
  }

  if(BootPos.bWiktionary) processFile(WIKTIONARY)
  if(BootPos.bUseTrainingData) processFile(TRAINING_DIR)

  def test = {
    println(language + ' ' + corpus);
    
    tagResults = new TaggingResult()
    val iter = getWordTagIteratorFromFile(TEST_DIR)
    val testData = new ArrayBuffer[Array[String]](10000)
    iter.copyToBuffer(testData)
    println("test tokens: " + testData.length)
    val results = tagger.test(testData)
    tagResults.processTaggingResults(results, testData, sentenceSepWord)

    tagResults.getAccuracy()
    // println("Most frequent tag overall: "+ tagger.bestTagsOverall)
    if(BootPos.bUniversalTags) println(tagMap.unmappedTags + " unmapped tags.")
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def getFileName(fileType: String): String = {
    if(fileType == WIKTIONARY)
      return DATA_DIR + WIKTIONARY
    var languageCorpusString = language;
    if(!corpus.equals(""))
      languageCorpusString += '/' + corpus
    var dir = DATA_DIR + languageCorpusString

    val subDir = fileType.replace("raw", "train")
    dir += '/'+ subDir + '/'
//     println("dir: " + dir)
//     println(new File(dir).list.toList)

    val files = new File(dir).list.toList.filter(_ contains fileType)
    val file = dir+files.head
    file
  }

/*  Purpose: While processing lines read from a file,
    replace empty lines with appropriate sentenceSeparator lines.
    If necessary, capitalize the line.*/
  def lineMap(newSentenceLine: String = sentenceSepWord)(x:String)= {
    var y = x.trim;
    if(y.isEmpty()) y= newSentenceLine;
    if(bIgnoreCase) y.map(_.toUpper)
    else y
  }

//    @return Iterator[Array[String]] whose elements are arrays of size 2, whose
//      first element is the word and second element is the corresponding tag.
//    Confidence in correctness: High
//    Reason: Used many times without problems.
  def getWordTagIteratorFromFile(mode: String): Iterator[Array[String]] = {
//      Determine wordField, tagField, sep
    val file = getFileName(mode)
    
    var wordField = 1
    var tagField = 3;
    var sep = '\t'
    if(language.equals("danish")) tagField = 4
    if(mode.equals(WIKTIONARY))tagField = 2
    else if(! (BootPos.conllCorpora contains corpus)) {
      wordField = 0; tagField = 1;
      if(List("", "hmm") contains corpus)
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
        Array(word, tag)
      })
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def processFile(mode: String) = {
    val iter = getWordTagIteratorFromFile(mode)
    val untaggedDataFile = getFileName("raw")
    val tokensUntagged = new ArrayBuffer[String]()

    if(bProcessUntaggedData){
      tokensUntagged ++= new TextTableParser(file = untaggedDataFile, encodingIn = encoding, lineMapFn = lineMap()).getColumn(0)
      if(tokensUntagged.head != sentenceSepWord) tokensUntagged prepend sentenceSepWord
      if(tokensUntagged.last != sentenceSepWord) tokensUntagged += sentenceSepWord
    }

    if(mode == WIKTIONARY) {
      // get words to consider.
      val testWords = getWordTagIteratorFromFile(TEST_DIR).map(_(0)).toSet
      val dict = new Dictionary(iter, testWords ++ tokensUntagged)
      dict.addEntry(sentenceSepWord, sentenceSepTag)
      dict.updateCompleteness(tokensUntagged)
      tagger.trainWithDictionary(dict)
    }
    else if(!bTrainingDataAsDictionary) tagger.train(iter)
    else {
      val dict = new Dictionary(iter)
      // dict.removeDuplicateEntries
      dict.updateCompleteness(tokensUntagged)
      tagger.trainWithDictionary(dict)
    }
    if(bProcessUntaggedData){
      tagger.processUntaggedData(tokensUntagged)
    }

  }
}

