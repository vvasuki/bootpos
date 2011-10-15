name := "BootPOS"

version := "0.1.1"

organization := "OpenNLP"

scalaVersion := "2.9.1"

crossPaths := false

retrieveManaged := true

resolvers += "opennlp sourceforge repo" at "http://opennlp.sourceforge.net/maven2"

libraryDependencies += "org.apache.opennlp" % "opennlp-tools" % "1.5.1-incubating"

libraryDependencies += "org.apache.opennlp" % "opennlp-maxent" % "3.0.1-incubating"

// For some reason the below are downloaded to lib_managed/jars but not to lib_managed/compile
libraryDependencies += "com.weiglewilczek.slf4s" %% "slf4s" % "[1.0,)" % "compile"

libraryDependencies += "ch.qos.logback" % "logback-classic" % "[0.9,)"

libraryDependencies += "ch.qos.logback" % "logback-core" % "[0.9,)"

