package opennlp.bootpos.util
class ReflectionUtil{
  def deepCopy[A](a: A)(implicit m: reflect.Manifest[A]): A =
    util.Marshal.load[A](util.Marshal.dump(a))
}
