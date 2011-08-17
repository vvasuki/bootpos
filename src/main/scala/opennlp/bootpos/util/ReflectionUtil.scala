package opennlp.bootpos.util
object reflectionUtil{
  def deepCopy[A](a: A)(implicit m: reflect.Manifest[A]): A =
    util.Marshal.load[A](util.Marshal.dump(a))
}
