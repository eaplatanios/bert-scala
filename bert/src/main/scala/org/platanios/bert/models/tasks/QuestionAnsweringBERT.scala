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

package org.platanios.bert.models.tasks

import org.platanios.bert.models.layers.BERT
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.io.CheckpointReader
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops.OpSpecification
import org.platanios.tensorflow.api.ops.variables.{Initializer, Regularizer, Reuse}

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class QuestionAnsweringBERT[T: TF : IsHalfOrFloatOrDouble](
    val config: QuestionAnsweringBERT.Config,
    val name: String = "QuestionAnsweringBERT"
) {
  protected val input: Input[QuestionAnsweringBERT.In] = {
    // Each batch contains:
    //   - `uniqueIDs`:  [batchSize]
    //   - `inputIDs`:   [batchSize, sequenceLength]
    //   - `inputMask`: [batchSize, sequenceLength]
    //   - `segmentIDs`: [batchSize, sequenceLength]
    Input(
      dataType = (INT64, INT32, INT32, INT32),
      shape = (Shape(-1), Shape(-1, -1), Shape(-1, -1), Shape(-1, -1)),
      name = "Input")
  }

  protected val trainInput: Input[QuestionAnsweringBERT.TrainIn] = {
    // Each batch contains:
    //   - `startPositions`: [batchSize]
    //   - `endPositions`:   [batchSize]
    Input(
      dataType = (INT32, INT32),
      shape = (Shape(-1), Shape(-1)),
      name = "TrainInput")
  }

  protected val estimator: QuestionAnsweringBERT.Estimator[T] = {
    tf.device("/GPU:0") {
      val optimizer = tf.train.AMSGrad(config.learningRate)
      val model = config.maxGradNorm match {
        case Some(norm) =>
          tf.learn.Model.simpleSupervised(
            input = input,
            trainInput = trainInput,
            layer = layer,
            loss = loss,
            optimizer = optimizer,
            clipGradients = tf.learn.ClipGradientsByGlobalNorm(norm),
            colocateGradientsWithOps = config.colocateGradientsWithOps)
        case None =>
          tf.learn.Model.simpleSupervised(
            input = input,
            trainInput = trainInput,
            layer = layer,
            loss = loss,
            optimizer = optimizer,
            colocateGradientsWithOps = config.colocateGradientsWithOps)
      }

      // Create estimator hooks.
      var hooks = Set[tf.learn.Hook]()

      // Add logging hooks.
      if (config.logLossFrequency > 0)
        hooks += tf.learn.LossLogger(log = true, trigger = StepHookTrigger(config.logLossFrequency))

      // Add summaries/checkpoints hooks.
      hooks ++= Set(
        tf.learn.StepRateLogger(log = false, summaryDir = config.summaryDir, trigger = StepHookTrigger(100)),
        tf.learn.SummarySaver(config.summaryDir, StepHookTrigger(config.summarySteps)),
        tf.learn.CheckpointSaver(config.workingDir, StepHookTrigger(config.checkpointSteps)))

      var sessionConfig = SessionConfig(
        allowSoftPlacement = Some(config.allowSoftPlacement),
        logDevicePlacement = Some(config.logDevicePlacement),
        gpuAllowMemoryGrowth = Some(config.gpuAllowMemoryGrowth))
      if (config.useXLA)
        sessionConfig = sessionConfig.copy(optGlobalJITLevel = Some(SessionConfig.L1GraphOptimizerGlobalJIT))

      // Create estimator.
      tf.learn.InMemoryEstimator(
        model, tf.learn.Configuration(
          workingDir = Some(config.workingDir),
          sessionConfig = Some(sessionConfig),
          randomSeed = config.randomSeed),
        trainHooks = hooks)
    }
  }

  def train(
      dataset: () => tf.data.Dataset[(QuestionAnsweringBERT.In, QuestionAnsweringBERT.TrainIn)],
      stopCriteria: tf.learn.StopCriteria = tf.learn.StopCriteria()
  ): Unit = {
    tf.device("/GPU:0") {
      estimator.train(dataset, stopCriteria)
    }
  }

  def predict(dataset: () => tf.data.Dataset[QuestionAnsweringBERT.In]): Iterator[Unit] = {
    estimator.infer(dataset).map {
      case (startLogits, endLogits) =>


        ???
    }
  }

  protected def bertLayer: BERT[T] = {
    new BERT[T](config.bertConfig, config.useOneHotEmbeddings)
  }

  protected def layer: Layer[QuestionAnsweringBERT.In, QuestionAnsweringBERT.Out[T]] = {
    new Layer[QuestionAnsweringBERT.In, QuestionAnsweringBERT.Out[T]](name) {
      override val layerType: String = "TrainLayer"

      override def forwardWithoutContext(
          input: QuestionAnsweringBERT.In
      )(implicit mode: Mode): QuestionAnsweringBERT.Out[T] = {
        tf.device("/GPU:0") {
          tf.learn.withParameterGetter(parameterGetter(this)) {
            val (uniqueIDs, inputIDs, inputMask, segmentIDs) = input

            // Encode the input sequences using BERT.
            val bertOutput = bertLayer.forwardWithoutContext(BERT.In(inputIDs, inputMask, segmentIDs))
            val sequenceOutput = bertOutput.encoderLayers.last

            // Use a linear layer for the span prediction.
            val logits = tf.variableScope("Classification") {
              val weights = getParameter[T](
                name = "Weights",
                shape = Shape(sequenceOutput.shape(-1), 2),
                initializer = BERT.createInitializer(config.bertConfig.initializerRange))
              val bias = getParameter[T](
                name = "Bias",
                shape = Shape(2),
                initializer = tf.ZerosInitializer)
              tf.linear(sequenceOutput, weights, bias)
            }

            val unstackedLogits = tf.unstack(tf.transpose(logits, Tensor(2, 0, 1)))
            val startLogits = unstackedLogits.head
            val endLogits = unstackedLogits.last
            (uniqueIDs, startLogits, endLogits)
          }
        }
      }
    }
  }

  protected def loss: Layer[(QuestionAnsweringBERT.Out[T], QuestionAnsweringBERT.TrainIn), Output[Float]] = {
    new Layer[(QuestionAnsweringBERT.Out[T], QuestionAnsweringBERT.TrainIn), Output[Float]](name) {
      override val layerType: String = "Loss"

      override def forwardWithoutContext(
          input: (QuestionAnsweringBERT.Out[T], QuestionAnsweringBERT.TrainIn)
      )(implicit mode: Mode): Output[Float] = {
        tf.device("/GPU:0") {
          val sequenceLength = tf.shape(input._1._2).slice(1)

          def computeLoss(logits: Output[Float], positions: Output[Int]): Output[Float] = {
            val oneHotPositions = tf.oneHot[Float, Int](positions, depth = sequenceLength)
            val logProbabilities = tf.logSoftmax(logits, axis = -1)
            -tf.mean(tf.sum(oneHotPositions * logProbabilities, axes = Tensor(-1)))
          }

          val startLoss = computeLoss(input._1._2.toFloat, input._2._1)
          val endLoss = computeLoss(input._1._3.toFloat, input._2._2)

          (startLoss + endLoss) / tf.constant(2.0f)
        }
      }
    }
  }

  protected def parameterGetter(layer: tf.learn.Layer[_, _]): tf.learn.ParameterGetter = {
    val checkpointReader = config.bertCheckpoint.map(CheckpointReader(_))
    val variables = checkpointReader.map(_.variableDataTypes).getOrElse(Map.empty)
    new tf.learn.ParameterGetter {
      override def get[P: TF](
          name: String,
          shape: Shape,
          initializer: Initializer,
          regularizer: Regularizer,
          trainable: Boolean,
          reuse: Reuse,
          collections: Set[Graph.Key[Variable[Any]]],
          cachingDevice: OpSpecification => String
      ): Output[P] = {
        def mapName(name: String): String = {
          var mappedName = name.substring("QuestionAnsweringBERT/".length)
          mappedName = mappedName.replaceAll("BERT", "bert")
          mappedName = mappedName.replaceAll("WordEmbeddings", "word_embeddings")
          mappedName = mappedName.replaceAll("TokenTypeEmbeddings", "token_type_embeddings")
          mappedName = mappedName.replaceAll("PositionEmbeddings", "position_embeddings")
          mappedName = mappedName.replaceAll("Embeddings", "embeddings")
          mappedName = mappedName.replaceAll("Encoder", "encoder")
          mappedName = mappedName.replaceAll("Pooler", "pooler/dense")
          mappedName = mappedName.replaceAll("Layer([0-9]+)", "layer_$1")
          mappedName = mappedName.replaceAll("Intermediate", "intermediate/dense")
          mappedName = mappedName.replaceAll("Output", "output")
          mappedName = mappedName.replaceAll("output/Weights", "output/dense/kernel")
          mappedName = mappedName.replaceAll("output/Bias", "output/dense/bias")
          mappedName = mappedName.replaceAll("Attention", "attention")
          mappedName = mappedName.replaceAll("Self", "self")
          mappedName = mappedName.replaceAll("Query", "query")
          mappedName = mappedName.replaceAll("Key", "key")
          mappedName = mappedName.replaceAll("Value", "value")
          mappedName = mappedName.replaceAll("Weights", "kernel")
          mappedName = mappedName.replaceAll("Bias", "bias")
          mappedName = mappedName.replaceAll("Beta", "beta")
          mappedName = mappedName.replaceAll("Gamma", "gamma")
          mappedName
        }

        val variableScope = tf.currentVariableScope.name
        val mappedName = mapName(if (variableScope != null && variableScope != "") s"$variableScope/$name" else name)
        (config.bertCheckpoint, variables.get(mappedName)) match {
          case (Some(path), Some(dataType)) =>
            tf.createWith(device = "/CPU:0", controlDependencies = Set.empty) {
              val value = Op.Builder[(Output[String], Output[String], Output[String]), Seq[Output[P]]](
                opType = "RestoreV2",
                name = name,
                input = (path.toAbsolutePath.toString, Tensor[String](mappedName), Tensor[String](""))
              ).setAttribute("dtypes", Array(dataType))
                  .build().output.head
              tf.variable[P](
                name, shape, tf.ConstantInitializer(value), regularizer, trainable, reuse, collections, cachingDevice)
            }
          case _ =>
            tf.variable[P](
              name, shape, initializer, regularizer, trainable, reuse, collections, cachingDevice)
        }

        // TODO: Find a way to use the "parent" parameter getter.
      }
    }
  }
}

object QuestionAnsweringBERT {
  type In = ( /* uniqueIDs */ Output[Long], /* inputIDs */ Output[Int], /* inputMask */ Output[Int], /* segmentIDs */ Output[Int])
  type TrainIn = ( /* startPositions */ Output[Int], /* endPositions */ Output[Int])
  type Out[T] = ( /* uniqueIDs */ Output[Long], /* startLogits */ Output[T], /* endLogits */ Output[T])

  type Estimator[T] = tf.learn.Estimator[
      /* In       */ In,
      /* TrainIn  */ (In, TrainIn),
      /* Out      */ Out[T],
      /* TrainOut */ Out[T],
      /* Loss     */ Float,
      /* EvalIn   */ (Out[T], (In, TrainIn))]

  case class Config(
      bertConfig: BERT.Config,
      bertCheckpoint: Option[Path],
      workingDir: Path,
      summaryDir: Path,
      useOneHotEmbeddings: Boolean = false,
      trainBatchSize: Int = 12,
      inferBatchSize: Int = 8,
      learningRate: Float = 3e-5f,
      maxGradNorm: Option[Float] = Some(1.0f),
      colocateGradientsWithOps: Boolean = true,
      checkpointSteps: Int = 1000,
      summarySteps: Int = 100,
      logLossFrequency: Int = 100,
      allowSoftPlacement: Boolean = true,
      logDevicePlacement: Boolean = false,
      gpuAllowMemoryGrowth: Boolean = false,
      useXLA: Boolean = false,
      randomSeed: Option[Int] = None)
}
