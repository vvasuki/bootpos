package opennlp.bootpos.util
 
class MathUtil{

// Addition of numbers represented in log space.
// Useful for adding probabilities.
  def logAdd(x: Double, y: Double): Double = {
    if(x == 0 || y == 0) return x+y
    if(x >= y) x + java.lang.Math.log1p(Math.exp(y-x))
    else y + java.lang.Math.log1p(Math.exp(x-y))
  }
}
