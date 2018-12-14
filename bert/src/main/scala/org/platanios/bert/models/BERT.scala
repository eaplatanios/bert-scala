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

package org.platanios.bert.models

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.io.CheckpointReader

import better.files._
import _root_.io.circe._
import _root_.io.circe.generic.auto._
import _root_.io.circe.generic.extras._
import _root_.io.circe.parser._
import _root_.io.circe.syntax._

/**
  * @author Emmanouil Antonios Platanios
  */
object BERT {
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

  def main(args: Array[String]): Unit = {
    val configFile = File("temp") / "models" / "uncased_L-12_H-768_A-12" / "bert_config.json"
    val ckptFile = File("temp") / "models" / "uncased_L-12_H-768_A-12" / "bert_model.ckpt"
    val config = Config.fromFile(configFile)
    val checkpointReader = CheckpointReader(ckptFile.path)
    val variables = checkpointReader.variableShapes
    variables.foreach(v => println(v._1))
    println("haha Christoph")
  }
}
