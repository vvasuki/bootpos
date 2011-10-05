name := "BootPOS"

version := "0.1.1"

organization := "OpenNLP"

scalaVersion := "2.9.1"

crossPaths := false

retrieveManaged := true

resolvers += "opennlp sourceforge repo" at "http://opennlp.sourceforge.net/maven2"

libraryDependencies += "org.apache.opennlp" % "opennlp-tools" % "1.5.1-incubating"

libraryDependencies += "org.apache.opennlp" % "opennlp-maxent" % "3.0.1-incubating"

libraryDependencies += "com.weiglewilczek.slf4s" %% "slf4s" % "1.0.7" % "compile"

// Maven repository seems to disallow wget style downloads.
// libraryDependencies += "ch.qos.logback" % "logback-classic" % "0.9.29" from "http://repo1.maven.org/maven2/ch/qos/logback/logback-classic/0.9.29/logback-classic-0.9.29.jar"
// libraryDependencies += "ch.qos.logback" % "logback-core" % "0.9.29" from "http://repo1.maven.org/maven2/ch/qos/logback/logback-classic/0.9.29/logback-core-0.9.29.jar"

