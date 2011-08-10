import sbt._

class BootPosProject (info: ProjectInfo) extends DefaultProject(info) {
  override def disableCrossPaths = true 

  // Add repositories
  val opennlpRepo = "opennlp sourceforge repo" at "http://opennlp.sourceforge.net/maven2"

  // Dependencies
  val opennlpTools = "org.apache.opennlp" % "opennlp-tools" % "1.5.1-incubating"
  val opennlpMaxent = "org.apache.opennlp" % "opennlp-maxent" % "3.0.1-incubating"
}
