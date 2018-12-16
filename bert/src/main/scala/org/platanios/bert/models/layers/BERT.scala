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

package org.platanios.bert.models.layers

import org.platanios.bert.models._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode

import better.files._
import _root_.io.circe._
import _root_.io.circe.generic.auto._
import _root_.io.circe.generic.extras._
import _root_.io.circe.parser._
import _root_.io.circe.syntax._

/** BERT model ("Bidirectional Embedding Representations from a Transformer").
  *
  * @param  config              Model configuration.
  * @param  useOneHotEmbeddings Boolean value indicating whether to add embeddings for the position of each token
  *                             in the input sequence.
  * @param  name                Name for this BERT model.
  *
  * @author Emmanouil Antonios Platanios
  */
class BERT[T: TF : IsHalfOrFloatOrDouble](
    val config: BERT.Config,
    val useOneHotEmbeddings: Boolean = false,
    override val name: String = "BERT"
) extends tf.learn.Layer[BERT.In, BERT.Out[T]](name) {
  override val layerType: String = "BERT"

  protected implicit val context: Context = Context(this)

  override def forwardWithoutContext(input: BERT.In)(implicit mode: Mode): BERT.Out[T] = {
    tf.variableScope("BERT") {
      val (embeddingsOutput, embeddingsTable) = tf.variableScope("Embeddings") {
        // Perform embedding lookup on the word IDs.
        val (embeddingsOutput, embeddingsTable) = Helpers.embeddingLookup(
          input.inputIDs,
          config.vocabularySize,
          config.hiddenSize,
          config.initializerRange,
          "WordEmbeddings",
          useOneHotEmbeddings)

        // Add positional embeddings and token type embeddings, and then layer normalize and perform dropout.
        val postProcessedEmbeddingsOutput = Helpers.embeddingPostProcessor(
          embeddingsOutput,
          tokenTypeIDs = Some(input.tokenTypeIDs),
          tokenTypeVocabularySize = config.typeVocabularySize,
          tokenTypeEmbeddingsTableName = "TokenTypeEmbeddings",
          usePositionEmbeddings = true,
          positionEmbeddingsTableName = "PositionEmbeddings",
          initializerRange = config.initializerRange,
          maxPositionEmbeddings = config.maxPositionEmbeddings,
          dropoutProbability = if (mode.isTraining) config.hiddenDropoutProbability else 0.0f)
        (postProcessedEmbeddingsOutput, embeddingsTable)
      }

      val encoderLayers = tf.variableScope("Encoder") {
        // This converts a 2D mask of shape [batchSize, sequenceLength] to a 3D mask of shape
        // [batchSize, sequenceLength, sequenceLength] which is used for the attention scores.
        val attentionMask = Helpers.createAttentionMaskFromInputMask(input.inputIDs, input.inputMask)

        // Create the stacked transformer.
        // `sequenceOutput` shape = [batchSize, sequenceLength, hiddenSize].
        Helpers.transformer(
          input = embeddingsOutput,
          attentionMask = Some(attentionMask),
          hiddenSize = config.hiddenSize,
          numHiddenLayers = config.numHiddenLayers,
          numAttentionHeads = config.numAttentionHeads,
          intermediateSize = config.intermediateSize,
          intermediateActivation = config.activation,
          hiddenDropoutProbability = if (mode.isTraining) config.hiddenDropoutProbability else 0.0f,
          attentionDropoutProbability = if (mode.isTraining) config.attentionDropoutProbability else 0.0f,
          initializerRange = config.initializerRange)
      }
//
//      val sequenceOutput = encoderLayers.last
//
//      // The "pooler" converts the encoded sequence tensor of shape [batchSize, sequenceLength, hiddenSize] to a tensor
//      // of shape [batchSize, hiddenSize]. This is necessary for segment-level (or segment-pair-level) classification
//      // tasks where we need a fixed dimensional representation of the segment.
//      val pooledOutput = tf.variableScope("Pooler") {
//        // We "pool" the model by simply taking the hidden state corresponding to the first token. We assume that this
//        // has been pre-trained.
//        val firstTokenTensor = sequenceOutput(::, 0, ::)
//        val weights = getParameter[T](
//          name = "Weights",
//          shape = Shape(firstTokenTensor.shape(-1), config.hiddenSize),
//          initializer = BERT.createInitializer(config.initializerRange))
//        val bias = getParameter[T](
//          name = "Bias",
//          shape = Shape(config.hiddenSize),
//          initializer = tf.ZerosInitializer)
//        Tanh(tf.linear(firstTokenTensor, weights, bias))
//      }

      BERT.Out(embeddingsOutput, encoderLayers, embeddingsTable)
    }
  }
}

object BERT {
  case class In(inputIDs: Output[Int], inputMask: Output[Int], tokenTypeIDs: Output[Int])

  case class Out[T: TF : IsHalfOrFloatOrDouble](
      embeddingsOutput: Output[T],
      encoderLayers: Seq[Output[T]],
      embeddingsTable: Output[T])

  /** JSON configuration used for serializing configurations. */
  private implicit val jsonConfig: Configuration = Configuration.default

  /** Configuration for the BERT model.
    *
    * @param  vocabularySize              Vocabulary size.
    * @param  hiddenSize                  Size of the encoder and the pooler layers.
    * @param  numHiddenLayers             Number of hidden layers in the encoder.
    * @param  numAttentionHeads           Number of attention heads for each attention layer in the encoder.
    * @param  intermediateSize            Size of the "intermediate" (i.e., feed-forward) layer in the encoder.
    * @param  activation                  Activation function used in the encoder and the pooler.
    * @param  hiddenDropoutProbability    Dropout probability for all fully connected layers in the embeddings, the
    *                                     encoder, and the pooler.
    * @param  attentionDropoutProbability Dropout probability for the attention scores.
    * @param  maxPositionEmbeddings       Maximum sequence length that this model might ever be used with. Typically,
    *                                     this is set to something large, just in case (e.g., 512, 1024, or 2048).
    * @param  typeVocabularySize          Vocabulary size for the token type IDs passed into the BERT model.
    * @param  initializerRange            Standard deviation of the truncated Normal initializer used for initializing
    *                                     all weight matrices.
    */
  @ConfiguredJsonCodec case class Config(
      @JsonKey("vocab_size") vocabularySize: Int,
      @JsonKey("hidden_size") hiddenSize: Int = 768,
      @JsonKey("num_hidden_layers") numHiddenLayers: Int = 12,
      @JsonKey("num_attention_heads") numAttentionHeads: Int = 12,
      @JsonKey("intermediate_size") intermediateSize: Int = 3072,
      @JsonKey("hidden_act") activation: Activation = GELU,
      @JsonKey("hidden_dropout_prob") hiddenDropoutProbability: Float = 0.1f,
      @JsonKey("attention_probs_dropout_prob") attentionDropoutProbability: Float = 0.1f,
      @JsonKey("max_position_embeddings") maxPositionEmbeddings: Int = 512,
      @JsonKey("type_vocab_size") typeVocabularySize: Int = 16,
      @JsonKey("initializer_range") initializerRange: Float = 0.02f
  ) {
    def writeToFile(file: File): Unit = {
      file.write(this.asJson.spaces2)
    }
  }

  object Config {
    @throws[Error]
    def fromFile(file: File): Config = {
      decode[Config](file.lines.mkString("\n")) match {
        case Left(error) => throw error
        case Right(config) => config
      }
    }
  }

  private implicit val encodeActivation: Encoder[Activation] = new Encoder[Activation] {
    final def apply(a: Activation): Json = Json.fromString(a.toString)
  }

  private implicit val decodeActivation: Decoder[Activation] = new Decoder[Activation] {
    final def apply(c: HCursor): Decoder.Result[Activation] = {
      c.as[String].map {
        case "linear" => Linear
        case "relu" => ReLU
        case "gelu" => GELU
        case "tanh" => Tanh
      }
    }
  }

  private[models] def createInitializer(range: Float = 0.02f): tf.VariableInitializer = {
    tf.RandomTruncatedNormalInitializer(standardDeviation = range)
  }
}
