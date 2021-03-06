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

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
object Helpers {
  //region Basic Manipulation

  def shapeDim[T: TF](tensor: Output[T], axis: Int): Output[Int] = {
    if (tensor.shape(axis) > -1)
      tf.constant(tensor.shape(axis))
    else
      tf.shape(tensor).slice(axis)
  }

  /** Reshapes a tensor with rank `>= 2` to a rank-2 tensor (i.e., a matrix).
    *
    * @param  input Tensor to reshape.
    * @return Tensor reshaped as a matrix.
    */
  def reshapeToMatrix[T: TF](input: Output[T]): Output[T] = {
    if (input.rank == 2)
      input
    else
      tf.reshape(input, Shape(-1, input.shape(-1)))
  }

  /** Reshapes a previously-reshaped rank-2 tensor back to its original shape.
    *
    * @param  input         Tensor to reshape.
    * @param  originalShape Original shape of tensor, to reshape back to.
    * @return Reshaped tensor.
    */
  def reshapeFromMatrix[T: TF](input: Output[T], originalShape: Output[Int]): Output[T] = {
    tf.reshape(input, tf.concatenate(Seq(originalShape(0 :: -1), tf.constant(input.shape(-1))(NewAxis))))
  }

  //endregion Basic Manipulation

  /** Perform dropout.
    *
    * @param  value              Value on which to apply dropout.
    * @param  dropoutProbability Dropout probability.
    * @return `value` after dropout has been applied.
    */
  def dropout[T: TF : IsHalfOrFloatOrDouble](value: Output[T], dropoutProbability: Float): Output[T] = {
    if (dropoutProbability == 0.0f)
      value
    else
      tf.dropout(value, 1.0f - dropoutProbability)
  }

  /** Applies layer normalization on the last dimension of `value`.
    *
    * @param  value Tensor to normalize.
    * @param  name  Name scope for the created ops.
    * @return `value` after normalization has been applied.
    */
  def layerNormalization[T: TF : IsDecimal](
      value: Output[T],
      name: String = "LayerNorm"
  )(implicit context: Context): Output[T] = {
    tf.variableScope(name) {
      val paramsShape = value.shape(-1 ::)

      // Allocate parameters for the beta and gamma of the normalization.
      val beta = context.layer.getParameter[T]("Beta", paramsShape, tf.ZerosInitializer)
      val gamma = context.layer.getParameter[T]("Gamma", paramsShape, tf.OnesInitializer)

      // Calculate the moments on the last axis (layer activations).
      val mean = tf.mean(value, axes = -1, keepDims = true)
      val variance = tf.mean(tf.square(value - mean), axes = -1, keepDims = true)

      // Compute layer normalization using the batch normalization function.
      val epsilon = tf.constant(1e-12f)
      val normalizedInput = (value - mean) * tf.rsqrt(variance + epsilon.castTo[T])
      val result = normalizedInput * gamma + beta
      result.setShape(value.shape)
      result
    }
  }

  /** Looks up word embeddings for the provided input IDs.
    *
    * @param  inputIDs                Tensor containing the word IDs to use for the lookup, with shape
    *                                 [batchSize, sequenceLength, numInputs] or [batchSize, sequenceLength].
    * @param  vocabularySize          Vocabulary size.
    * @param  embeddingSize           Word embeddings size.
    * @param  initializerRange        Embeddings table initializer standard deviation.
    * @param  wordEmbeddingsTableName Name of the embedding table.
    * @param  useOneHotEmbeddings     If `true`, the one-hot method is used for word embeddings. Otherwise, a `gather`
    *                                 op is used. One hot is better for TPUs.
    * @return Tuple containing the embedded IDs (with shape [batchSize, sequenceLength, embeddingSize]), along with the
    *         embeddings table.
    */
  def embeddingLookup[T: TF : IsNotQuantized](
      inputIDs: Output[Int],
      vocabularySize: Int,
      embeddingSize: Int = 128,
      initializerRange: Float = 0.02f,
      wordEmbeddingsTableName: String = "WordEmbeddings",
      useOneHotEmbeddings: Boolean = false
  )(implicit context: Context): (Output[T], Output[T]) = {
    // This function assumes that the input is of shape [batchSize, sequenceLength, numInputs].
    // If the input is a two-dimensional tensor of shape [batchSize, sequenceLength],
    // we reshape it to [batchSize, sequenceLength, 1].
    val reshapedInputIDs = {
      if (inputIDs.rank == 2) {
        tf.expandDims(inputIDs, axis = Tensor(-1))
      } else {
        inputIDs
      }
    }

    val embeddingTable = context.layer.getParameter[T](
      name = wordEmbeddingsTableName,
      shape = Shape(vocabularySize, embeddingSize),
      initializer = BERT.createInitializer(initializerRange))

    val result = {
      if (useOneHotEmbeddings) {
        val flatInputIDs = tf.reshape(reshapedInputIDs, Shape(-1))
        val oneHotInputIDs = tf.oneHot(flatInputIDs, depth = vocabularySize)
        tf.matmul(oneHotInputIDs, embeddingTable)
      } else {
        tf.embeddingLookup(embeddingTable, reshapedInputIDs)
      }
    }

    val inputShape = tf.shape(reshapedInputIDs)
    val resultShape = tf.concatenate(Seq(inputShape(0 :: -1), inputShape(-1, NewAxis) * embeddingSize))
    val reshapedResult = tf.reshape(result, resultShape)
    reshapedResult.setShape(reshapedInputIDs.shape(0 :: -1) ++ Shape(reshapedInputIDs.shape(-1) * embeddingSize))

    (reshapedResult, embeddingTable)
  }

  /** Performs post-processing on the `input` embeddings tensor.
    *
    * The post-processing includes optionally adding token type and position embeddings.
    *
    * @param  input                        Input embeddings tensor.
    * @param  tokenTypeIDs                 Token type IDs tensor, with shape [batchSize, sequenceLength].
    * @param  tokenTypeVocabularySize      Vocabulary size for the the token type IDs.
    * @param  tokenTypeEmbeddingsTableName Name of the token type embeddings table.
    * @param  usePositionEmbeddings        Boolean value indicating whether to add embeddings for the position of each
    *                                      token in the input sequence.
    * @param  positionEmbeddingsTableName  Name of the position embeddings table.
    * @param  initializerRange             Embeddings table initializer standard deviation.
    * @param  maxPositionEmbeddings        Maximum sequence length that might ever be used with this model. This can be
    *                                      longer than the sequence length of the input tensor, but cannot be shorter.
    * @param  dropoutProbability           Dropout probability applied to the final output tensor.
    * @return Post-processed embeddings table.
    */
  def embeddingPostProcessor[T: TF : IsHalfOrFloatOrDouble](
      input: Output[T],
      tokenTypeIDs: Option[Output[Int]],
      tokenTypeVocabularySize: Int,
      tokenTypeEmbeddingsTableName: String,
      usePositionEmbeddings: Boolean,
      positionEmbeddingsTableName: String,
      initializerRange: Float,
      maxPositionEmbeddings: Int,
      dropoutProbability: Float
  )(implicit context: Context): Output[T] = {
    val inputShape = tf.shape(input)
    val batchSize = if (input.shape(0) > -1) tf.constant(input.shape(0)) else inputShape(0)
    val sequenceLength = if (input.shape(1) > -1) tf.constant(input.shape(1)) else inputShape(1)
    val width = input.shape(2)

    var result = input

    // Add token type embeddings, if needed.
    tokenTypeIDs match {
      case None => ()
      case Some(ids) =>
        val tokenTypeEmbeddingsTable = context.layer.getParameter[T](
          name = tokenTypeEmbeddingsTableName,
          shape = Shape(tokenTypeVocabularySize, width),
          initializer = BERT.createInitializer(initializerRange))
        // This vocabulary will be small and so we always use the one-hot approach here, because it is always faster
        // for small vocabularies.
        val flatTokenTypeIDs = tf.reshape(ids, Shape(-1))
        val oneHotIDs = tf.oneHot(flatTokenTypeIDs, depth = tokenTypeVocabularySize)
        val tokenTypeEmbeddings = tf.matmul(oneHotIDs, tokenTypeEmbeddingsTable)
        result += tf.reshape(tokenTypeEmbeddings, tf.stack(Seq(batchSize, sequenceLength, tf.constant(width))))
    }

    if (usePositionEmbeddings) {
      tf.createWith(controlDependencies = Set(tf.assertLessEqual(sequenceLength, maxPositionEmbeddings))) {
        val fullPositionEmbeddings = context.layer.getParameter[T](
          name = positionEmbeddingsTableName,
          shape = Shape(maxPositionEmbeddings, width),
          initializer = BERT.createInitializer(initializerRange))
        // Since the position embeddings table is a learned variable, we create it using a (long) sequence length,
        // `maxPositionEmbeddings`. The actual sequence length might be shorter than this, for faster training of tasks
        // that do not have long sequences. So, `fullPositionEmbeddings` is effectively an embedding table for positions
        // [0, 1, 2, ..., maxPositionEmbeddings - 1], and the current sequence has positions
        // [0, 1, 2, ..., sequenceLength - 1], so we can just perform a slice.
        val positionEmbeddings = tf.slice(
          fullPositionEmbeddings,
          begin = Seq(0, 0),
          size = tf.stack(Seq(sequenceLength, tf.constant(-1))))

        // Only the last two dimensions are relevant (`sequenceLength` and `width`), and so we broadcast among the
        // first dimensions, which is typically just the batch size.
        val one = tf.constant(1)
        val positionBroadcastShape = Seq.fill(result.rank - 2)(one) ++ Seq(sequenceLength, tf.constant(width))
        val reshapedPositionEmbeddings = tf.reshape(positionEmbeddings, tf.stack(positionBroadcastShape))
        result += reshapedPositionEmbeddings
      }
    }

    result = layerNormalization(result)
    result = dropout(result, dropoutProbability)
    result
  }

  /** Creates a 3D attention mask from a 2D tensor mask.
    *
    * @param  fromTensor 2D or 3D tensor with shape [batchSize, fromSequenceLength, ...].
    * @param  toMask     Tensor with shape [batchSize, toSequenceLength].
    * @return Tensor with shape [batchSize, toSequenceLength].
    */
  def createAttentionMaskFromInputMask[T: TF : IsNotQuantized](
      fromTensor: Output[T],
      toMask: Output[Int]
  ): Output[Float] = {
    val inputShape = tf.shape(fromTensor)
    val batchSize = if (fromTensor.shape(0) > -1) tf.constant(fromTensor.shape(0)) else inputShape(0)
    val fromSequenceLength = if (fromTensor.shape(1) > -1) tf.constant(fromTensor.shape(1)) else inputShape(1)

    val toSequenceLength = if (toMask.shape(1) > -1) tf.constant(toMask.shape(1)) else inputShape(1)
    val reshapedToMask = tf.reshape(toMask, tf.stack(Seq(batchSize, tf.constant(1), toSequenceLength))).toFloat

    // We do not assume that `fromTensor` is a mask (although it could be). We do not actually care if we attend *from*
    // padding tokens (only *to* padding tokens) so we create a tensor of all ones.
    val broadcastOnes = tf.ones[Float](tf.stack(Seq(batchSize, fromSequenceLength, tf.constant(1))))

    // Here we broadcast along two dimensions to create the mask.
    broadcastOnes * reshapedToMask
  }

  /** Performs multi-headed attention from `from_Tensor` to `toTensor`.
    *
    * This is an implementation of multi-headed attention based on "Attention is all you Need". If `fromTensor` and
    * `toTensor` are the same, then this is self-attention. Each sequence step in `fromTensor` attends to the
    * corresponding sequence in `toTensor`, and returns a fixed-width vector.
    *
    * This function first projects `fromTensor` into a "query" tensor and `toTensor` into "key" and "value" tensors.
    * These are (effectively) a list of tensors of length `numHeads`, where each tensor is of shape
    * `[batchSize, sequenceLength, sizePerHead]`.
    *
    * Then, it performs a dot product between the query and they key tensors and scales them. These are softmaxed to
    * obtain attention probabilities. The value tensors are then interpolated by these probabilities, and then
    * concatenated back to a single tensor and returned.
    *
    * In practice, the multi-headed attention is done with transpose and reshape operations, rather than with separate
    * tensors.
    *
    * @param  fromTensor         Tensor with shape `[batchSize, fromSequenceLength, fromWidth]`.
    * @param  toTensor           Tensor with shape `[batchSize, toSequenceLength, toWidth]`.
    * @param  attentionMask      Tensor with shape [batchSize, fromSequenceLength, toSequenceLength]`. The
    *                            values should be `1` or `0`. The attention scores will effectively be set to
    *                            negative infinity for any positions in the mask that are `0`, and will be
    *                            unchanged for positions that are `1`.
    *                            @param  numHeads                    Number of attention heads.
    *                            @param  sizePerHead                 Size of each attention head.
    *                            @param  queryActivation             Activation function for the attention queries.
    *                            @param  keyActivation               Activation function for the attention keys.
    *                            @param  valueActivation             Activation function for the attention values.
    *                            @param  attentionDropoutProbability Dropout probability for the attention scores.
    *                            @param  initializerRange            Embeddings table initializer standard deviation.
    *                            @param  returnMatrix                If true`, the output will be of shape
    *                            `[batchSize * fromSequenceLength, numHeads * sizePerHead]`. If `false`, the
    *                            output will be of shape
    *                            `[batchSize, fromSequenceLength, numHeads * sizePerHead]`.
    * @param  batchSize          If the input is 2D, then this might be the batch size of the 3D version
    *                            of the `fromTensor` and `toTensor`.
    * @param  fromSequenceLength If the input is 2D, then this might be the sequence length of the 3D version
    *                            of the `fromTensor`.
    * @param  toSequenceLength   If the input is 2D, then this might be the sequence length of the 3D version
    *                            of the `toTensor`.
    * @return Tensor with shape `[batchSize, fromSequenceLength, numHeads * sizePerHead]`. If `returnMatrix` is `true`,
    *         this will be of shape `[batchSize * fromSequenceLength, numHeads * sizePerHead]`).
    * @throws IllegalArgumentException If the shapes of `fromTensor` and `toTensor` are invalid.
    */
  @throws[IllegalArgumentException]
  def multiHeadAttention[T: TF : IsHalfOrFloatOrDouble](
      fromTensor: Output[T],
      toTensor: Output[T],
      attentionMask: Option[Output[Float]] = None,
      numHeads: Int = 1,
      sizePerHead: Int = 512,
      queryActivation: Activation = Linear,
      keyActivation: Activation = Linear,
      valueActivation: Activation = Linear,
      attentionDropoutProbability: Float = 0.0f,
      initializerRange: Float = 0.02f,
      returnMatrix: Boolean = false,
      batchSize: Option[Output[Int]] = None,
      fromSequenceLength: Option[Output[Int]] = None,
      toSequenceLength: Option[Output[Int]] = None
  )(implicit context: Context): Output[T] = {
    if (fromTensor.rank != toTensor.rank)
      throw new IllegalArgumentException("The rank of 'fromTensor' must match that of 'toTensor'.")

    if (fromTensor.rank == 2 && (batchSize.isEmpty || fromSequenceLength.isEmpty || toSequenceLength.isEmpty)) {
      throw new IllegalArgumentException(
        "When passing in rank-2 tensors, the values for 'batchSize', 'fromSequenceLength', " +
            "and 'toSequenceLength' must all be provided.")
    }

    // Scalar dimensions referenced here:
    //   - B = batch size (number of sequences)
    //   - F = `fromTensor` sequence length
    //   - T = `toTensor` sequence length
    //   - N = number of attention heads
    //   - H = size per head
    val B = batchSize.getOrElse(shapeDim(fromTensor, 0))
    val F = fromSequenceLength.getOrElse(shapeDim(fromTensor, 1))
    val T = toSequenceLength.getOrElse(shapeDim(toTensor, 1))
    val N = tf.constant(numHeads)
    val H = tf.constant(sizePerHead)

    val fromTensor2D = reshapeToMatrix(fromTensor)
    val toTensor2D = reshapeToMatrix(toTensor)

    // Query = [B*F, N*H]
    val queryWeights = context.layer.getParameter[T](
      name = "Query/Weights",
      shape = Shape(fromTensor2D.shape(-1), numHeads * sizePerHead),
      initializer = BERT.createInitializer(initializerRange))
    val queryBias = context.layer.getParameter[T](
      name = "Query/Bias",
      shape = Shape(numHeads * sizePerHead),
      initializer = tf.ZerosInitializer)
    val query = queryActivation(tf.linear(fromTensor2D, queryWeights, queryBias))

    // Key = [B*T, N*H]
    val keyWeights = context.layer.getParameter[T](
      name = "Key/Weights",
      shape = Shape(toTensor2D.shape(-1), numHeads * sizePerHead),
      initializer = BERT.createInitializer(initializerRange))
    val keyBias = context.layer.getParameter[T](
      name = "Key/Bias",
      shape = Shape(numHeads * sizePerHead),
      initializer = tf.ZerosInitializer)
    val key = keyActivation(tf.linear(toTensor2D, keyWeights, keyBias))

    // Value = [B*T, N*H]
    val valueWeights = context.layer.getParameter[T](
      name = "Value/Weights",
      shape = Shape(toTensor2D.shape(-1), numHeads * sizePerHead),
      initializer = BERT.createInitializer(initializerRange))
    val valueBias = context.layer.getParameter[T](
      name = "Value/Bias",
      shape = Shape(numHeads * sizePerHead),
      initializer = tf.ZerosInitializer)
    val value = valueActivation(tf.linear(toTensor2D, valueWeights, valueBias))

    def transposeForScores(
        input: Output[T],
        batchSize: Output[Int],
        numHeads: Output[Int],
        sequenceLength: Output[Int],
        width: Output[Int]
    ): Output[T] = {
      val reshaped = tf.reshape(input, tf.stack(Seq(batchSize, sequenceLength, numHeads, width)))
      tf.transpose(reshaped, Tensor(0, 2, 1, 3))
    }

    // Transposed Query = [B, N, F, H]
    val transposedQuery = transposeForScores(query, B, N, F, H)

    // Transposed Key = [B, N, T, H]
    val transposedKey = transposeForScores(key, B, N, T, H)

    // We take the dot product between "query" and "key" to get the raw attention scores.
    // Attention Scores = [B, N, F, T]
    var attentionScores = tf.matmul(transposedQuery, transposedKey, transposeB = true)
    attentionScores /= tf.constant(math.sqrt(sizePerHead)).castTo[T]

    attentionMask.foreach(mask => {
      // `attentionMask` = [B, 1, F, T]
      val reshapedMask = tf.expandDims(mask, Tensor(1))

      // Since the attention mask is 1.0 for positions we want to attend and 0.0 for masked positions, this operation
      // will create a tensor which is 0.0 for positions we want to attend and -10000.0 for masked positions.
      val adder = (1.0f - reshapedMask.toFloat) * -10000.0f

      // Since we are adding it to the raw scores, before the softmax,
      // this is effectively the same as removing this entirely.
      attentionScores += adder.castTo[T]
    })

    // Normalize the attention scores to probabilities.
    // Attention Probabilities = [B, N, F, T]
    var attentionProbabilities = tf.softmax(attentionScores)

    // This is actually dropping out entire tokens to attend to, which might seem a bit unusual, but is taken from the
    // original Transformer paper.
    attentionProbabilities = dropout(attentionProbabilities, attentionDropoutProbability)

    // Transposed Value = [B, N, T, H]
    val transposedValue = transposeForScores(value, B, N, T, H)

    // Output = [B, N, F, H]
    var output = tf.matmul(attentionProbabilities, transposedValue)

    // Output = [B, F, N, H]
    output = tf.transpose(output, Tensor(0, 2, 1, 3))

    if (returnMatrix) {
      // Output = [B*F, N*H]
      output = tf.reshape(output, tf.stack(Seq(B * F, N * H)))
    } else {
      // Output = [B, F, N*H]
      output = tf.reshape(output, tf.stack(Seq(B, F, N * H)))
    }

    output
  }

  /** Creates a multi-headed, multi-layer Transformer from "Attention is All You Need".
    *
    * This is almost an exact implementation of the original Transformer encoder. Refer to the
    * [original paper](https://arxiv.org/abs/1706.03762) and
    * [implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
    * for details.
    *
    * @param  input                       Tensor with shape `[batchSize, sequenceLength, inputWidth]`.
    * @param  attentionMask               Tensor with shape `[batchSize, fromSequenceLength, toSequenceLength]`. The
    *                                     values should be `1` or `0`. The attention scores will effectively be set to
    *                                     negative infinity for any positions in the mask that are `0`, and will be
    *                                     unchanged for positions that are `1`.
    * @param  hiddenSize                  Hidden size of the Transformer.
    * @param  numHiddenLayers             Number of layers (blocks) in the Transformer.
    * @param  numAttentionHeads           Number of attention heads.
    * @param  intermediateSize            Size of the "intermediate" (a.k.a., feed forward) layer.
    * @param  intermediateActivation      Activation function to apply to the output of the "intermediate" (a.k.a.,
    *                                     feed forward) layer.
    * @param  hiddenDropoutProbability    Dropout probability to use for the hidden layers.
    * @param  attentionDropoutProbability Dropout probability to use for the attention scores.
    * @param  initializerRange            Embeddings table initializer standard deviation.
    * @return Sequence containing the Transformer layer outputs (one tensor per layer).
    * @throws IllegalArgumentException If the hidden size is not a multiple of the number of attention heads.
    */
  @throws[IllegalArgumentException]
  def transformer[T: TF : IsHalfOrFloatOrDouble](
      input: Output[T],
      attentionMask: Option[Output[Float]],
      hiddenSize: Int,
      numHiddenLayers: Int,
      numAttentionHeads: Int,
      intermediateSize: Int,
      intermediateActivation: Activation,
      hiddenDropoutProbability: Float,
      attentionDropoutProbability: Float,
      initializerRange: Float
  )(implicit context: Context): Seq[Output[T]] = {
    if (hiddenSize % numAttentionHeads != 0) {
      throw new IllegalArgumentException(
        s"The hidden size ($hiddenSize) is not a multiple of the number of attention heads ($numAttentionHeads).")
    }

    val attentionHeadSize = hiddenSize / numAttentionHeads
    val batchSize = shapeDim(input, 0)
    val sequenceLength = shapeDim(input, 1)

    // The Transformer performs sum residuals on all layers so that the input
    // needs to have the same depth as the hidden size.
    if (input.shape(2) > -1 && input.shape(2) != hiddenSize) {
      throw new IllegalArgumentException(
        s"The width of the input tensor (${input.shape(2)}) is different than the hidden size ($hiddenSize).")
    }

    // We keep the representation as a 2D tensor to avoid reshaping it back and forth from a 3D tensor to a 2D tensor.
    // Reshapes are normally free on the GPU/CPU but may not be free on the TPU, and so we want to minimize them to
    // help the optimizer.
    var previousOutput = reshapeToMatrix(input)

    val allLayerOutputs = Array.ofDim[Output[T]](numHiddenLayers)
    for (layerIndex <- 0 until numHiddenLayers) {
      tf.variableScope(s"Layer$layerIndex") {
        val attentionOutput = tf.variableScope("Attention") {
          var attentionOutput = tf.variableScope("Self") {
            multiHeadAttention(
              fromTensor = previousOutput,
              toTensor = previousOutput,
              attentionMask = attentionMask,
              numHeads = numAttentionHeads,
              sizePerHead = attentionHeadSize,
              attentionDropoutProbability = attentionDropoutProbability,
              initializerRange = initializerRange,
              returnMatrix = true,
              batchSize = Some(batchSize),
              fromSequenceLength = Some(sequenceLength),
              toSequenceLength = Some(sequenceLength))
          }

          // Run a linear projection of `hiddenSize` and then add a residual connection to `previousOutput`.
          tf.variableScope("Output") {
            val weights = context.layer.getParameter[T](
              name = "Weights",
              shape = Shape(attentionOutput.shape(-1), hiddenSize),
              initializer = BERT.createInitializer(initializerRange))
            val bias = context.layer.getParameter[T](
              name = "Bias",
              shape = Shape(hiddenSize),
              initializer = tf.ZerosInitializer)
            attentionOutput = tf.linear(attentionOutput, weights, bias)
            attentionOutput = dropout(attentionOutput, hiddenDropoutProbability)
            attentionOutput = layerNormalization(attentionOutput + previousOutput)
          }

          attentionOutput
        }

        // The activation is only applied to the "intermediate" hidden layer.
        val intermediateOutput = tf.variableScope("Intermediate") {
          val weights = context.layer.getParameter[T](
            name = "Weights",
            shape = Shape(attentionOutput.shape(-1), intermediateSize),
            initializer = BERT.createInitializer(initializerRange))
          val bias = context.layer.getParameter[T](
            name = "Bias",
            shape = Shape(intermediateSize),
            initializer = tf.ZerosInitializer)
          val intermediateOutput = tf.linear(attentionOutput, weights, bias)
          intermediateActivation(intermediateOutput)
        }

        // Down-project back to `hiddenSize` and add the residual.
        tf.variableScope("Output") {
          val weights = context.layer.getParameter[T](
            name = "Weights",
            shape = Shape(intermediateOutput.shape(-1), hiddenSize),
            initializer = BERT.createInitializer(initializerRange))
          val bias = context.layer.getParameter[T](
            name = "Bias",
            shape = Shape(hiddenSize),
            initializer = tf.ZerosInitializer)
          var layerOutput = tf.linear(intermediateOutput, weights, bias)
          layerOutput = dropout(layerOutput, hiddenDropoutProbability)
          layerOutput = layerNormalization(layerOutput + attentionOutput)
          previousOutput = layerOutput
          allLayerOutputs(layerIndex) = layerOutput
        }
      }
    }

    allLayerOutputs.map(output => {
      val reshaped = reshapeFromMatrix(output, tf.shape(input))
      reshaped.setShape(input.shape(0 :: -1) ++ Shape(output.shape(-1)))
      reshaped
    })
  }
}
