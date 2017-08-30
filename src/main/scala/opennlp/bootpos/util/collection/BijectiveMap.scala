package opennlp.bootpos.util.collection

import scala.collection.mutable.HashMap

/*
 *Extends the concept of a hash map to be bijective.
 *So, bijective maps have two additional properties:
 *a] Values are unique.
 *b] Looking up keys based on values is an O(1) time operation.
 */
class BijectiveHashMap[A, B] extends HashMap[A, B]{
  private var mapBA = new HashMap[B, A]

//  Confidence in correctness: High.
//  Reason: Well tested.
  override def put(key: A, value: B): Option[B] = {
    if(super.contains(key))
      throw new IllegalArgumentException("Key already exists, remove key first.")
    if(mapBA.contains(value))
      throw new IllegalArgumentException("Value already exists, remove value first.")
    mapBA.put(value, key)
    super.put(key, value)
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  override def remove(key: A): Option[B] = {
    super.remove(key) match {
      case Some(x) => {mapBA.remove(x); return Some(x)}
      case None => return None
    }
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def getKey(value: B): Option[A] = {
    return mapBA.get(value)
  }

//  Confidence in correctness: High.
//  Reason: Well tested.
  def removeValue(value: B): Option[A] = {
    mapBA.remove(value) match{
      case Some(x) => {super.remove(x); return Some(x)}
      case None => return None
    }
  }


}
