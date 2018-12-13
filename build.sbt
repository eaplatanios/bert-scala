/* Copyright 2018, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

import ReleaseTransformations._
import sbtrelease.Vcs

import scala.sys.process.Process

scalaVersion in ThisBuild := "2.12.7"
crossScalaVersions in ThisBuild := Seq("2.11.12", "2.12.7")

organization in ThisBuild := "org.platanios"

// In order to update the snapshots more frequently, the Coursier "Time-To-Live" (TTL) option can be modified. This can
// be done by modifying the "COURSIER_TTL" environment variable. Its value is parsed using
// 'scala.concurrent.duration.Duration', so that things like "24 hours", "5 min", "10s", or "0s", are fine, and it also
// accepts infinity ("Inf") as a duration. It defaults to 24 hours, meaning that the snapshot artifacts are updated
// every 24 hours.
resolvers in ThisBuild += Resolver.sonatypeRepo("snapshots")

val tensorFlowScalaVersion = "0.4.2-SNAPSHOT"

autoCompilerPlugins in ThisBuild := true

scalacOptions in ThisBuild ++= Seq(
  "-deprecation",
  "-encoding", "UTF-8",
  "-feature",
  "-language:existentials",
  "-language:higherKinds",
  "-language:implicitConversions",
  "-unchecked",
  "-Yno-adapted-args",
  "-Xfuture")

val scalacProfilingEnabled: SettingKey[Boolean] =
  settingKey[Boolean]("Flag specifying whether to enable profiling for the Scala compiler.")

scalacProfilingEnabled in ThisBuild := false

lazy val loggingSettings = Seq(
  libraryDependencies ++= Seq(
    "com.typesafe.scala-logging" %% "scala-logging"   % "3.9.0",
    "ch.qos.logback"             %  "logback-classic" % "1.2.3"))

lazy val commonSettings = loggingSettings ++ Seq(
  // Plugin that prints better implicit resolution errors.
  // addCompilerPlugin("io.tryp"  % "splain" % "0.3.3" cross CrossVersion.patch)
)

lazy val testSettings = Seq(
  libraryDependencies ++= Seq(
    "junit"         %  "junit"     % "4.12",
    "org.scalactic" %% "scalactic" % "3.0.4",
    "org.scalatest" %% "scalatest" % "3.0.4" % "test"),
  logBuffered in Test := false,
  fork in test := false,
  testForkedParallel in Test := false,
  parallelExecution in Test := false,
  testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest, "-oDF"))

lazy val tensorFlowSettings = Seq(
  libraryDependencies += "org.platanios" %% "tensorflow" % tensorFlowScalaVersion, // classifier "darwin-cpu-x86_64",
)

lazy val all = (project in file("."))
    .aggregate(bert)
    .dependsOn(bert)
    .settings(moduleName := "bert", name := "BERT")
    .settings(commonSettings)
    .settings(publishSettings)
    .settings(
      assemblyJarName in assembly := s"bert-${version.value}.jar",
      // mainClass in assembly := Some("org.platanios.bert.experiments.Experiment"),
      test in assembly := {},
      sourcesInBase := false,
      unmanagedSourceDirectories in Compile := Nil,
      unmanagedSourceDirectories in Test := Nil,
      unmanagedResourceDirectories in Compile := Nil,
      unmanagedResourceDirectories in Test := Nil,
      publishArtifact := true)

lazy val bert = (project in file("./bert"))
    .settings(moduleName := "bert", name := "BERT")
    .settings(commonSettings)
    .settings(testSettings)
    .settings(tensorFlowSettings)
    .settings(publishSettings)
    .settings(
      libraryDependencies ++= Seq(
        "com.github.pathikrit" %% "better-files" % "3.4.0",
        "org.apache.commons" % "commons-compress" % "1.16.1",
        "com.twitter" %% "util-collection" % "18.11.0"))
