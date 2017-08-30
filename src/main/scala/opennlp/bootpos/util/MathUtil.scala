package opennlp.bootpos.util
 
object mathUtil{

// Addition of numbers represented in log space.
// Useful for adding probabilities.
  def logAdd(x: Double, y: Double): Double = {
    if(x == math.log(0)) return y
    if(y == math.log(0)) return x
    if(x >= y) x + java.lang.Math.log1p(math.exp(y-x))
    else y + java.lang.Math.log1p(math.exp(x-y))
  }

}
