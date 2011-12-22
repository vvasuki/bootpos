package opennlp.bootpos.app

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import opennlp.bootpos.util.collection._
import opennlp.bootpos.util.io.TextTableParser
import opennlp.bootpos.tag._
import opennlp.bootpos.tag.hmm._
import java.util.NoSuchElementException
import org.slf4j.LoggerFactory
import java.io.File

class TagMap(TAG_MAP_DIR: String, languageCode: String, corpus: String, sentenceSepTagIn: String = null) {
  val log = LoggerFactory.getLogger(this.getClass)
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
    case e: java.io.FileNotFoundException => log warn ("Alert: no tag map found!" + e)
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

    var iter = tagsUniversal.filterNot(x => (x.equals("X") || x.equals(".")))
    for(tagUniversal <- iter) {
      if(tag.indexOf(tagUniversal) != -1) {
        tagMap(tag) = tagUniversal; return
      }
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
    log info ("unknown tag "+tag + " :word "+ word);
    tagMap(tag) = "X";
    log info tag + " -> " + tagMap(tag)
  }
  
//  Confidence in correctness: High.
//  Reason: Proved.
  def getMappedTag(tagIn1: String, word: String): String = {
    var tag = tagIn1.map(_.toUpper)
    try{tag = tagMap(tag);}
    catch{
      case e => {updateTagMap(tag, word); tag = tagMap(tag)}
    }
    return tag
  }

  override def toString = tagMap.toString

}

