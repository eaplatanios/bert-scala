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

package org.platanios.bert.data

import org.platanios.bert.data.tokenization._
import org.platanios.bert.models.layers.BERT
import org.platanios.bert.models.tasks.QuestionAnsweringBERT
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.io.TFRecordWriter
import org.platanios.tensorflow.api.ops.Parsing.FixedLengthFeature

import better.files._
import com.typesafe.scalalogging.Logger
import _root_.io.circe.generic.auto._
import _root_.io.circe.parser._
import org.slf4j.LoggerFactory
import org.tensorflow.example._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
object SQuAD {
  private val logger = Logger(LoggerFactory.getLogger("Data / SQuAD"))

  case class Span(start: Int, end: Int) {
    def length: Int = {
      end - start + 1
    }
  }

  /** A single example for simple sequence classification. */
  case class Example(
      questionID: String,
      question: String,
      documentTokens: Seq[String],
      originalAnswer: Option[String] = None,
      answerSpan: Option[Span] = None,
      isImpossible: Boolean = false)

  /** A single set of data features. */
  case class Features(
      uniqueID: Long,
      exampleIndex: Int,
      docSpanIndex: Int,
      tokens: Seq[String],
      tokenToOriginalMap: Map[Int, Int],
      tokenIsMaxContext: Map[Int, Boolean],
      inputIDs: Seq[Long],
      inputMask: Seq[Boolean],
      segmentIDs: Seq[Long],
      answerSpan: Option[Span] = None,
      isImpossible: Boolean = false)

  /** Reads a SQuAD Json file into a sequence of SQuAD [[Example]]s.
    *
    * @param  isTraining Boolean flag indicating whether the examples being loaded are training examples.
    * @param  file       Json file that contains the examples.
    * @return Sequence of loaded SQuAD examples.
    * @throws IllegalArgumentException If `isTraining` is `true` and there exists a question with zero or more than one
    *                                  answers.
    */
  @throws[IllegalArgumentException]
  def readExamples(isTraining: Boolean, file: File): Seq[Example] = {
    decode[JsonData](file.lines.mkString("\n")) match {
      case Left(error) => throw error
      case Right(data) =>
        val version = data.version
        data.data.flatMap(document => {
          document.paragraphs.flatMap(paragraph => {
            val paragraphText = paragraph.context
            val docTokens = mutable.ArrayBuffer[String]()
            var previousIsWhitespace = true
            val charToWordOffset = paragraphText.map(c => {
              if (isWhitespace(c)) {
                previousIsWhitespace = true
              } else {
                if (previousIsWhitespace)
                  docTokens.append(c.toString)
                else
                  docTokens.update(docTokens.length - 1, docTokens.last + c)
                previousIsWhitespace = false
              }
              docTokens.length - 1
            })

            paragraph.qas.flatMap(qa => {
              val questionID = qa.id
              val question = qa.question
              val isImpossible = isTraining && version == "v2.0" && qa.is_impossible.getOrElse(false)

              var originalAnswer: Option[String] = None
              var answerSpan: Option[Span] = None
              var skipQuestion: Boolean = false
              if (isTraining && !isImpossible) {
                if (qa.answers.length != 1)
                  throw new IllegalArgumentException("For training, each question should have exactly 1 answer.")
                val answer = qa.answers.head
                val answerOffset = answer.answer_start
                val answerLength = answer.text.length
                val startPosition = charToWordOffset(answerOffset)
                val endPosition = charToWordOffset(answerOffset + answerLength - 1)
                originalAnswer = Some(answer.text)
                answerSpan = Some(Span(startPosition, endPosition))

                // We only add answers where the text can be exactly recovered from the document.
                // If this cannot happen it is likely due to weird Unicode stuff and so we just skip the example.
                // Note that this means that for the training mode, not every example is guaranteed to be preserved.
                val actualText = docTokens.slice(startPosition, endPosition + 1).mkString(" ")
                val cleanedAnswerText = whitespaceRegex.split(originalAnswer.get).mkString(" ")
                if (!actualText.contains(cleanedAnswerText))
                  skipQuestion = true
              }

              if (skipQuestion)
                None
              else
                Some(Example(questionID, question, docTokens, originalAnswer, answerSpan, isImpossible))
            })
          })
        })
    }
  }

  /** Creates an iterator over [[Features]], from the provided examples.
    *
    * @param  isTraining        Boolean flag indicating whether the examples being loaded are training examples.
    * @param  examples          Examples to convert to features.
    * @param  tokenizer         Tokenizer to use.
    * @param  maxSequenceLength Maximum allowed sequence length.
    * @param  documentStride    Document stride to use when splitting up a long document into chunks.
    * @param  maxQueryLength    Maximum number of tokens for the question. Questions longer than this will be truncated.
    * @param  logFirstN         Number of processed features to log, while going through the provided examples.
    * @return Iterator over created features.
    */
  def convertToFeatures(
      isTraining: Boolean,
      examples: Seq[Example],
      tokenizer: Tokenizer,
      maxSequenceLength: Int = 384,
      documentStride: Int = 128,
      maxQueryLength: Int = 64,
      logFirstN: Int = 20
  ): Iterator[Features] = {
    logger.info("Converting examples to features.")

    var progress = 0L
    var progressLogTime = System.currentTimeMillis

    var uniqueID = 1000000000L
    val features = examples.toIterator.zipWithIndex.flatMap {
      case (example, exampleIndex) =>
        val queryTokens = tokenizer.tokenize(example.question).take(maxQueryLength)

        // Map tokens and create reverse indices.
        var tokenToOriginalIndex = Seq.empty[Int]
        val originalToTokenIndex = Array.ofDim[Int](example.documentTokens.length)
        var allDocumentTokens = Seq.empty[String]
        example.documentTokens.zipWithIndex.foreach {
          case (token, tokenIndex) =>
            originalToTokenIndex(tokenIndex) = allDocumentTokens.length
            tokenizer.tokenize(token).foreach(subToken => {
              tokenToOriginalIndex :+= tokenIndex
              allDocumentTokens :+= subToken
            })
        }

        var tokenStartPosition = -1
        var tokenEndPosition = -1
        if (isTraining && !example.isImpossible) {
          tokenStartPosition = originalToTokenIndex(example.answerSpan.get.start)
          if (example.answerSpan.get.end < example.documentTokens.length - 1)
            tokenEndPosition = originalToTokenIndex(example.answerSpan.get.end + 1) - 1
          else
            tokenEndPosition = allDocumentTokens.length - 1
          val improvedSpan = improveAnswerSpan(
            allDocumentTokens, Span(tokenStartPosition, tokenEndPosition), tokenizer, example.originalAnswer.get)
          tokenStartPosition = improvedSpan.start
          tokenEndPosition = improvedSpan.end
        }

        // The -3 accounts for [CLS], [SEP], and [SEP].
        val maxDocumentTokens = maxSequenceLength - queryTokens.length - 3

        // We can have documents that are longer than the maximum sequence length. To deal with this we use a sliding
        // window approach, where we take chunks of the up to our max length with a stride of `documentStride`.
        var documentSpans = Seq.empty[Span]
        var startOffset = 0
        var continue = true
        while (continue) {
          val length = math.min(allDocumentTokens.length - startOffset, maxDocumentTokens)
          documentSpans :+= Span(startOffset, startOffset + length - 1)
          continue &&= startOffset + length < allDocumentTokens.length
          startOffset += math.min(length, documentStride)
        }

        val features = documentSpans.zipWithIndex.map {
          case (documentSpan, documentSpanIndex) =>
            var tokens = Seq.empty[String]
            var tokenToOriginalMap = Map.empty[Int, Int]
            var tokenIsMaxContext = Map.empty[Int, Boolean]
            var segmentIDs = Seq.empty[Long]

            // First, we add the [CLS] token, the question tokens, and the [SEP] token.
            tokens :+= "[CLS]"
            segmentIDs :+= 0L
            queryTokens.foreach(token => {
              tokens :+= token
              segmentIDs :+= 0L
            })
            tokens :+= "[SEP]"
            segmentIDs :+= 0L

            // Then, we add the current document span tokens, along with the [SEP] token.
            for (i <- 0 to documentSpan.end - documentSpan.start) {
              val splitTokenIndex = documentSpan.start + i
              tokenToOriginalMap += tokens.length -> tokenToOriginalIndex(splitTokenIndex)
              tokenIsMaxContext += tokens.length -> isMaxContext(documentSpans, documentSpanIndex, splitTokenIndex)
              tokens :+= allDocumentTokens(splitTokenIndex)
              segmentIDs :+= 1L
            }
            tokens :+= "[SEP]"
            segmentIDs :+= 1L

            var inputIDs = tokenizer.convertTokensToIDs(tokens)

            // This mask contains 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            var inputMask = Seq.fill(inputIDs.length)(true)

            // Zero-pad up to the sequence length.
            while (inputIDs.length < maxSequenceLength) {
              inputIDs :+= 0L
              inputMask :+= false
              segmentIDs :+= 0L
            }

            val answerSpan = {
              if (isTraining && example.isImpossible) {
                Some(Span(0, 0))
              } else if (isTraining) {
                // For training, if our document chunk does not contain an annotation we throw it out,
                // because there is nothing to predict.
                val docStart = documentSpan.start
                val docEnd = documentSpan.end
                val outOfSpan = tokenStartPosition < docStart || tokenEndPosition > docEnd
                if (outOfSpan) {
                  Some(Span(0, 0))
                } else {
                  val docOffset = queryTokens.length + 2
                  Some(Span(tokenStartPosition - docStart + docOffset, tokenEndPosition - docStart + docOffset))
                }
              } else {
                None
              }
            }

            if (exampleIndex < logFirstN) {
              val tokToOriginal = tokenToOriginalMap.map(p => s"${p._1}: ${p._2}").mkString(", ")
              val tokMaxContext = tokenIsMaxContext.map(p => s"${p._1}: ${p._2}").mkString(", ")
              logger.info(s"*** Example ***")
              logger.info(s"\t Unique ID:            $uniqueID")
              logger.info(s"\t Example ID:           $exampleIndex")
              logger.info(s"\t Document Span ID:     $documentSpanIndex")
              logger.info(s"\t Tokens:               ${tokens.mkString(" ")}")
              logger.info(s"\t Token-To-Original:    {$tokToOriginal}")
              logger.info(s"\t Token-Is-Max-Context: {$tokMaxContext}")
              logger.info(s"\t Input IDs:            [${inputIDs.mkString(", ")}]")
              logger.info(s"\t Input Mask:           [${inputMask.mkString(", ")}]")
              logger.info(s"\t Segment IDs:          [${segmentIDs.mkString(", ")}]")
              if (isTraining && example.isImpossible)
                logger.info("\t Impossible Example")
              if (isTraining && !example.isImpossible) {
                val answer = tokens.slice(answerSpan.get.start, answerSpan.get.end + 1).mkString(" ")
                logger.info(s"\t Answer Start:         ${answerSpan.get.start}")
                logger.info(s"\t Answer End:           ${answerSpan.get.end}")
                logger.info(s"\t Answer:               $answer")
              }
            }

            val features = Features(
              uniqueID, exampleIndex, documentSpanIndex, tokens,
              tokenToOriginalMap, tokenIsMaxContext, inputIDs,
              inputMask, segmentIDs, answerSpan, example.isImpossible)
            uniqueID += 1
            features
        }

        progress += 1L
        val time = System.currentTimeMillis
        if (time - progressLogTime >= 1e4) {
          val numBars = Math.floorDiv(progress, examples.length).toInt
          logger.info(
            s"│${"═" * numBars}${" " * (10 - numBars)}│ " +
                s"%${features.length.toString.length}s / ${examples.length} examples converted.".format(progress))
          progressLogTime = time
        }

        features
    }

    logger.info("Converted examples to features.")

    features
  }

  /** Writes the provided features as TensorFlow records in the provided file.
    *
    * @param  isTraining Boolean flag indicating whether the examples being loaded are training examples.
    * @param  features   Features to write as TensorFlow records.
    * @param  file       File in which to write the TensorFlow records.
    * @return Number of TensorFlow records written.
    */
  def writeFeatures(
      isTraining: Boolean,
      features: Iterator[Features],
      file: File,
      randomSeed: Option[Long] = None
  ): Int = {
    var numFeaturesWritten = 0
    val writer = TFRecordWriter(file.path)

    // If training, shuffle the features so that we can use a small shuffle buffer size when feeding data to the model.
    val shuffledFeatures = {
      if (!isTraining) {
        features.toArray
      } else {
        def fisherYatesShuffle[T](values: Array[T]): Array[T] = {
          val random = randomSeed.map(new scala.util.Random(_)).getOrElse(new scala.util.Random())
          values.indices.foreach(n => {
            val randomIndex = n + random.nextInt(values.length - n)
            val temp = values(randomIndex)
            values.update(randomIndex, values(n))
            values(n) = temp
          })
          values
        }

        val featuresArray = features.toArray
        fisherYatesShuffle(featuresArray)
      }
    }

    def intFeature(values: Seq[Long]): Feature = {
      Feature.newBuilder()
          .setInt64List(Int64List.newBuilder().addAllValue(values.map(long2Long).asJava))
          .build()
    }

    shuffledFeatures.foreach(feature => {
      val features = org.tensorflow.example.Features.newBuilder()
      features.putFeature("unique_ids", intFeature(Seq(feature.uniqueID)))
      features.putFeature("input_ids", intFeature(feature.inputIDs))
      features.putFeature("input_mask", intFeature(feature.inputMask.map(v => if (v) 1L else 0L)))
      features.putFeature("segment_ids", intFeature(feature.segmentIDs))

      if (isTraining) {
        features.putFeature("start_positions", intFeature(Seq(feature.answerSpan.get.start)))
        features.putFeature("end_positions", intFeature(Seq(feature.answerSpan.get.end)))
        features.putFeature("is_impossible", intFeature(Seq(if (feature.isImpossible) 1L else 0L)))
      }

      writer.write(org.tensorflow.example.Example.newBuilder().setFeatures(features).build())
      numFeaturesWritten += 1
    })
    writer.flush()
    writer.close()
    numFeaturesWritten
  }

  /** Returns tokenized answer spans that better match the annotated answer.
    *
    * The SQuAD annotations are character based. We first project them to whitespace-tokenized words. But then after
    * WordPiece tokenization, we can often find a "better match". For example:
    *
    * Question: What year was John Smith born?
    * Context: The leader was John Smith (1895-1943).
    * Answer: 1895
    *
    * The original whitespace-tokenized answer will be "(1895-1943).". However after tokenization, our tokens will be
    * "( 1895 - 1943 ) .". So we can match the exact answer, 1895.
    *
    * However, this is not always possible. Consider the following:
    *
    * Question: What country is the top exporter of electornics?
    * Context: The Japanese electronics industry is the lagest in the world.
    * Answer: Japan
    *
    * In this case, the annotator chose "Japan" as a character sub-span of the word "Japanese". Since our WordPiece
    * tokenizer does not split "Japanese", we just use "Japanese" as the annotation. This is fairly rare in SQuAD,
    * but does happen.
    *
    * @param  documentTokens Sequence containing all document/paragraph tokens.
    * @param  span           Answer span to improve.
    * @param  tokenizer      Tokenizer being used.
    * @param  originalAnswer Original answer.
    * @return Improved answer span that may be the same as the provided one.
    */
  private def improveAnswerSpan(
      documentTokens: Seq[String],
      span: Span,
      tokenizer: Tokenizer,
      originalAnswer: String
  ): Span = {
    val tokenizedOriginalAnswer = tokenizer.tokenize(originalAnswer).mkString(" ")
    val answers = (span.start to span.end)
        .flatMap(newS => {
          (span.end to newS by -1).map(newE => {
            (documentTokens.slice(newS, newE + 1).mkString(" "), Span(newS, newE))
          })
        })
        .filter(_._1 == tokenizedOriginalAnswer)
    if (answers.isEmpty)
      span
    else
      answers.head._2
  }

  /** Check if this is the "maximum context" document span for the token.
    *
    * Because of the sliding window approach taken to scoring documents, a single token can appear in multiple
    * documents. For example:
    *
    * Document: the man went to the store and bought a gallon of milk
    * Span 1: the man went to the
    * Span 2: to the store and bought
    * Span 3: and bought a gallon of
    * ...
    *
    * The word "bought" will have two scores, from spans 2 and 3. We only want to consider the score with
    * "maximum context", which we define as the *minimum* of its left and right context (the *sum* of left and right
    * context will always be the same, of course).
    *
    * In the example, the maximum context for "bought" would be span 3 because it has 1 left context and 3 right
    * context, while span 2 has 4 left context and 0 right context.
    *
    * @param  documentSpans    All document spans.
    * @param  currentSpanIndex Current document span index.
    * @param  tokenPosition    Position of the token in the current span.
    * @return Boolean value indicating whether this is the "maximum context" document span for the token.
    */
  private def isMaxContext(
      documentSpans: Seq[Span],
      currentSpanIndex: Int,
      tokenPosition: Int
  ): Boolean = {
    var bestScore = 0.0f
    var bestSpanIndex = 0
    documentSpans.zipWithIndex.foreach {
      case (documentSpan, index) =>
        if (tokenPosition >= documentSpan.start && tokenPosition <= documentSpan.end) {
          val numLeftContext = tokenPosition - documentSpan.start
          val numRightContext = documentSpan.end - tokenPosition
          val score = math.min(numLeftContext, numRightContext) + 0.01f * documentSpan.length
          if (score > bestScore) {
            bestScore = score
            bestSpanIndex = index
          }
        }
    }

    currentSpanIndex == bestSpanIndex
  }

  def createTrainDataset(
      file: File,
      batchSize: Int,
      numParallelCalls: Int
  ): tf.data.Dataset[(QuestionAnsweringBERT.In, QuestionAnsweringBERT.TrainIn)] = {
    tf.device("/CPU:0") {
      tf.data.datasetFromTFRecordFiles(file.path.toAbsolutePath.toString)
          .repeat()
          .shuffle(100)
          .mapAndBatch(
            function = record => {
              val parsed = parseTrainTFRecord(record)
              (parsed._1, (parsed._2._1, parsed._2._2))
            },
            batchSize = batchSize,
            numParallelCalls = numParallelCalls,
            dropRemainder = false)
    }
  }

  protected def parseTrainTFRecord(
      serialized: Output[String]
  ): ((Output[Long], Output[Int], Output[Int], Output[Int]), (Output[Int], Output[Int], Output[Int])) = {
    val example = tf.parseSingleExample(
      serialized = serialized,
      features = (
          FixedLengthFeature[Long](key = "unique_ids", shape = Shape()),
          FixedLengthFeature[Long](key = "input_ids", shape = Shape(-1)),
          FixedLengthFeature[Long](key = "input_mask", shape = Shape(-1)),
          FixedLengthFeature[Long](key = "segment_ids", shape = Shape(-1)),
          FixedLengthFeature[Long](key = "start_positions", shape = Shape()),
          FixedLengthFeature[Long](key = "end_positions", shape = Shape()),
          FixedLengthFeature[Long](key = "is_impossible", shape = Shape())),
      name = "ParsedTrainExample")
    ((example._1, example._2.toInt, example._3.toInt, example._4.toInt),
        (example._5.toInt, example._6.toInt, example._7.toInt))
  }

  protected def parseInferTFRecord(
      serialized: Output[String]
  ): (Output[Long], Output[Int], Output[Int], Output[Int]) = {
    val example = tf.parseSingleExample(
      serialized = serialized,
      features = (
          FixedLengthFeature[Long](key = "unique_ids", shape = Shape()),
          FixedLengthFeature[Long](key = "input_ids", shape = Shape(-1)),
          FixedLengthFeature[Long](key = "input_mask", shape = Shape(-1)),
          FixedLengthFeature[Long](key = "segment_ids", shape = Shape(-1))),
      name = "ParsedInferExample")
    (example._1, example._2.toInt, example._3.toInt, example._4.toInt)
  }

  private val whitespaceRegex: Regex = "\\s+".r

  private def isWhitespace(c: Char): Boolean = {
    c.isWhitespace || c == '\u202f'
  }

  //region Json Parsing Case Classes

  private case class JsonData(version: String, data: Seq[JsonDocument])
  private case class JsonDocument(title: String, paragraphs: Seq[JsonParagraph])
  private case class JsonParagraph(qas: Seq[JsonQuestion], context: String)
  private case class JsonQuestion(
      question: String,
      id: String,
      answers: Seq[JsonAnswer],
      plausible_answers: Option[Seq[JsonAnswer]],
      is_impossible: Option[Boolean])
  private case class JsonAnswer(text: String, answer_start: Int)

  //endregion Json Parsing Case Classes

  def main(args: Array[String]): Unit = {
    // === Pre-trained models ===
    // Each .zip file contains three items:
    //   - A TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
    //   - A vocab file (vocab.txt) to map WordPiece to word id.
    //   - A config file (bert_config.json) which specifies the hyperparameters of the model.
    //
    // BERT-Base, Uncased: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    //   12-layer, 768-hidden, 12-heads, 110M parameters
    // BERT-Large, Uncased: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
    //   24-layer, 1024-hidden, 16-heads, 340M parameters
    // BERT-Base, Cased: https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
    //   12-layer, 768-hidden, 12-heads , 110M parameters
    // BERT-Large, Cased: https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
    //   24-layer, 1024-hidden, 16-heads, 340M parameters
    // BERT-Base, Multilingual Cased (New, recommended): https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
    //   104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    // BERT-Base, Multilingual Uncased (Orig, not recommended): https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip
    //   102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    // BERT-Base, Chinese: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    //   Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

    val trainFile = "temp" / "data" / "squad" / "train-v2.0.json"
    val devFile = "temp" / "data" / "squad" / "dev-v2.0.json"
    val trainTFRecordsFile = "temp" / "data" / "squad" / "train.tfrecords"
    val devTFRecordsFile = "temp" / "data" / "squad" / "dev.tfrecords"

    val vocabFile = "temp" / "models" / "uncased_L-12_H-768_A-12" / "vocab.txt"
    val bertConfigFile = "temp" / "models" / "uncased_L-12_H-768_A-12" / "bert_config.json"
    val bertCkptFile = "temp" / "models" / "uncased_L-12_H-768_A-12" / "bert_model.ckpt"

    val vocabulary = Vocabulary.fromFile(vocabFile)
    val tokenizer = new FullTokenizer(vocabulary, caseSensitive = false, unknownToken = "[UNK]", maxWordLength = 200)

    if (trainTFRecordsFile.notExists) {
      writeFeatures(
        isTraining = true,
        features = convertToFeatures(isTraining = true, readExamples(isTraining = true, trainFile), tokenizer),
        file = trainTFRecordsFile,
        randomSeed = Some(12345L))
    }

    if (devTFRecordsFile.notExists) {
      writeFeatures(
        isTraining = false,
        features = convertToFeatures(isTraining = false, readExamples(isTraining = false, devFile), tokenizer),
        file = devTFRecordsFile,
        randomSeed = Some(12345L))
    }

    val trainDataset = () => createTrainDataset(trainTFRecordsFile, batchSize = 12, numParallelCalls = 4)
    val config = QuestionAnsweringBERT.Config(
      bertConfig = BERT.Config.fromFile(bertConfigFile),
      bertCheckpoint = Some(bertCkptFile.path),
      workingDir = ("temp" / "working-dirs" / "uncased_L-12_H-768_A-12").path,
      summaryDir = ("temp" / "summaries" / "uncased_L-12_H-768_A-12").path)
    val model = new QuestionAnsweringBERT[Float](config)
    model.train(trainDataset)
    println("haha")
  }
}
