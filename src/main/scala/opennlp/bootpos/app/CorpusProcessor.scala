package opennlp.bootpos.app

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import opennlp.bootpos.util.collection.BijectiveHashMap
import opennlp.bootpos.util.io.TextTableParser
import opennlp.bootpos.tag._
import opennlp.bootpos.tag.hmm._
import java.util.NoSuchElementException

class CorpusProcessor(language: String, corpus: String, taggerType: String = "WordTagProbabilities"){
  var numTestTokensKnown = 0
  var numTestTokensNovel = 0
  var correctTaggingsKnown = 0
  var correctTaggingsNovel = 0

  val DATA_DIR = BootPos.DATA_DIR
  val TEST_DIR = "test"
  val TRAINING_DIR = "train"
  val WIKTIONARY = DATA_DIR+"TEMP-S20110618.tsv"

  var unmappedTags = 0
  val LANGUAGE_CODE_MAP = getClass.getResource("/lang/languageCodes.properties").getPath
  val TAG_MAP_DIR = DATA_DIR+"universal_pos_tags.1.02/"

  val sentenceSepTag = "###"
  val sentenceSepWord = "###"




  var languageCode = language
  // Deduce language code
  var parser = new TextTableParser(file = LANGUAGE_CODE_MAP, separator = ' ', x =>x.length >= 2, lineMapFn = null)
  var iter = parser.getRowIterator.filter((x) => x(1).equalsIgnoreCase(language))
  if(iter.hasNext)
    languageCode = (iter.next())(0)
  println(languageCode)
  
//  Confidence in correctness: High.
//  Reason: Proved correct.
  var tagger: Tagger = null
  taggerType match {
    case "OpenNLP" => tagger = new OpenNLP( languageCode, sentenceSepTag, sentenceSepWord)
    case "HMM" => tagger = new HMM(sentenceSepTag, sentenceSepWord)
    case "LabelPropagation" => tagger = new LabelPropagationTagger(sentenceSepTag, sentenceSepWord)
    case _ => tagger = new WordTagProbabilities(sentenceSepTag, sentenceSepWord)
  }


  val tagMap = new HashMap[String, String]()
  //X - other: foreign words, typos, abbreviations
  //ADP - adpositions (prepositions and postpositions)
  //. - punctuation
  val tagsUniversal = List("VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", ".")

//  Confidence in correctness: High.
//  Reason: Well tested.
  if(BootPos.bUniversalTags)  {
    // Populate the tagMap by reading the appropriate file.
    parser = new TextTableParser(file = TAG_MAP_DIR + languageCode + "-" + corpus + ".map", filterFnIn = (x =>x.length >= 2), lineMapFn = (x => x.map(_.toUpper)))
    parser.getRowIterator.foreach(x => tagMap(x(0)) = x(1))
    tagMap.values.foreach(x => tagMap(x) = x)

  //  Add the universal tags themselves to the map.
    tagsUniversal.foreach(x => tagMap(x) = x)
//    print(tagMap)
  }

  if(BootPos.bWiktionary) processFile(WIKTIONARY)
  if(BootPos.bUseTrainingData) processFile(TRAINING_DIR)



//  Confidence in correctness: High.
//  Reason: Proved.
  def getMappedTag(tagIn1: String, word: String): String = {
    /*
     * Add mapping for a tag to the tag map.
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
  //    Begin special cases
  //ADP - adpositions (prepositions and postpositions)
      if(tag.indexOf("POSITION") != -1) {tagMap(tag) = "ADP"; return}
  //X - other: foreign words, typos, abbreviations
      if(tag.indexOf("ABBREV") != -1 || tag.indexOf("FOREIGN") != -1 || tag.indexOf("ACRONYM") != -1 || tag.indexOf("INITIAL") != -1) {tagMap(tag) = "X"; return}
  //. - punctuation
      if(tag.indexOf("PUNCTUAT") != -1) {tagMap(tag) = "X"; return}
      if(tag.indexOf("PARTICLE") != -1) {tagMap(tag) = "PRT"; return}
      if(tag.indexOf("ARTICLE") != -1) {tagMap(tag) = "DET"; return}
      unmappedTags = unmappedTags+1
      println("unmapped tag "+tag + " :word "+ word);
      tagMap(tag) = tag;
    }

    var tag = tagIn1
    try{tag = tagMap(tag);}
    catch{case e => updateTagMap(tag, word)}
    return tag
  }


//  Confidence in correctness: High.
//  Reason: Well tested.
  def getAccuracy() = {
    processFile(TEST_DIR)
    var correctTaggings = correctTaggingsKnown + correctTaggingsNovel
    var numTestTokens = numTestTokensKnown + numTestTokensNovel
    var accuracy = correctTaggings/ numTestTokens.toDouble
    var accuracyKnown = correctTaggingsKnown/ numTestTokensKnown.toDouble
    var accuracyNovel = correctTaggingsNovel/ numTestTokensNovel.toDouble

    printf("Accuracy: %.3f, (Known: %.3f, Novel: %.3f)\n", accuracy, accuracyKnown, accuracyNovel)
    printf("Non training tokens: %d, %.3f\n", numTestTokensNovel, numTestTokensNovel/numTestTokens.toDouble)
  }

  def test = {
    println(language + ' ' + corpus);
    getAccuracy()
    println("Most frequent tag overall: "+ tagger.bestTagsOverall)
    if(BootPos.bUniversalTags) println(unmappedTags + " unmapped tags.")
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def processFile(mode: String) = {

    def getFileName(fileType: String): String = {
      var languageCorpusString = language;
      if(!corpus.equals("")) languageCorpusString = languageCorpusString + '/' + corpus
      var file = DATA_DIR + languageCorpusString + '/'+ fileType+'/' + languageCorpusString.replace("/", "_") + '_'+ fileType
      if(!corpus.equals("")) file = file + ".conll"
      else file = file.replace("_", "")
      file
    }

    if(mode.equals(TEST_DIR)) {
      numTestTokensKnown = 0
      numTestTokensNovel = 0
      correctTaggingsKnown = 0
      correctTaggingsNovel = 0
    }

    val file = getFileName(mode)
//    println(file)

//    @return Iterator[Array[String]] whose elements are arrays of size 2, whose
//      first element is the word and second element is the corresponding tag.
//    Confidence in correctness: High
//    Reason: Used many times without problems.
    def getWordTagIteratorFromFile: Iterator[Array[String]] = {
//      Determine wordField, tagField, sep
      var wordField = 1
      var tagField = 3;
      var sep = '\t'
      if(language.equals("danish")) tagField = 4
      if(corpus.equals("")) {
        wordField = 0; tagField = 1; sep = '/'
      }
      if(mode.equals(WIKTIONARY))tagField = 2


//      Prepare a function to map empty lines to an empty sentence word/ token pair.
      var newSentenceLine = sentenceSepWord;
      for(i <- 1 to tagField) newSentenceLine = newSentenceLine + sep + sentenceSepTag
      var lineMap = (x:String)=> {var y = x.trim;
                                  if(y.isEmpty()) y= newSentenceLine;
                                  if(taggerType == "OpenNLP") y
                                  else y.map(_.toUpper)
                                  }

//      Prepare a function to filter the lines from the stream based on whether they have the right number of fields and language-tags.
      var filterFn = ((x:Array[String]) => (x.length >= tagField+1))
      if(mode.equals(WIKTIONARY))
        filterFn = ((x:Array[String]) => ((x.length >= tagField+1) && x(0).equalsIgnoreCase(language)))

      val parser = new TextTableParser(file = file, separator = sep, filterFnIn = filterFn, lineMapFn = lineMap)
      parser.getFieldIterator(wordField, tagField).map(x => {
          var tag = x(1); var word = x(0);
          if(BootPos.bUniversalTags) tag = getMappedTag(tag, word)
          Array(word, tag)
        })
    }

    val iter = getWordTagIteratorFromFile

    if(!mode.equals(TEST_DIR)) {
      tagger.train(iter)
    }
    else {
      val testData = new ArrayBuffer[Array[String]](10000)
      iter.copyToBuffer(testData)
      println(testData.length)
      val resultPair = tagger.predict(testData)
      for {i <- testData.indices.iterator
        if(testData(i)(0) != sentenceSepWord)
      }{
        val tag = testData(i)(0);
        val bCorrect  = resultPair(i)(0)
        val bNovelToken = resultPair(i)(1)
        
        if(!bNovelToken) {
          if(bCorrect) correctTaggingsKnown  = correctTaggingsKnown  + 1
          numTestTokensKnown = numTestTokensKnown + 1;
        }
        else {
          if(bCorrect) correctTaggingsNovel  = correctTaggingsNovel  + 1
          numTestTokensNovel = numTestTokensNovel + 1;
        }
      }
    }

  }

}
