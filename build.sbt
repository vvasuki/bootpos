name := "BootPOS"

version := "0.1.1"

organization := "OpenNLP"

scalaVersion := "2.9.1"

crossPaths := false

retrieveManaged := true

resolvers += "opennlp sourceforge repo" at "http://opennlp.sourceforge.net/maven2"

libraryDependencies += "org.apache.opennlp" % "opennlp-tools" % "1.5.1-incubating"

libraryDependencies += "org.apache.opennlp" % "opennlp-maxent" % "3.0.1-incubating"

