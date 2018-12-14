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

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait Activation {
  def apply[T: TF : IsReal](value: Output[T]): Output[T]
  override def toString: String
}

case object Linear extends Activation {
  override def apply[T: TF : IsReal](value: Output[T]): Output[T] = {
    value
  }

  override def toString: String = "linear"
}

case object ReLU extends Activation {
  override def apply[T: TF : IsReal](value: Output[T]): Output[T] = {
    tf.relu(value)
  }

  override def toString: String = "relu"
}

/** Gaussian Error Linear Unit.
  *
  * This is a smoother version of the ReLU, originally proposed in
  * [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415).
  */
case object GELU extends Activation {
  override def apply[T: TF : IsReal](value: Output[T]): Output[T] = {
    val half = tf.constant(0.5).castTo[T]
    val one = tf.constant(1.0).castTo[T]
    val sqrtTwo = tf.constant(math.sqrt(2.0)).castTo[T]
    val cdf = half * (one + tf.erf(value / sqrtTwo))
    value * cdf
  }

  override def toString: String = "gelu"
}

case object Tanh extends Activation {
  override def apply[T: TF : IsReal](value: Output[T]): Output[T] = {
    tf.tanh(value)
  }

  override def toString: String = "tanh"
}
