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

package org.platanios.bert.data.tokenization

import org.platanios.bert.data.newReader

import better.files._

/**
  * @author Emmanouil Antonios Platanios
  */
class Vocabulary(private val tokensToIDs: Map[String, Long]) {
  private val idsToTokens: Map[Long, String] = tokensToIDs.map(_.swap)

  def contains(token: String): Boolean = tokensToIDs.contains(token)
  def tokenToID(token: String): Long = tokensToIDs(token)
  def idToToken(id: Long): String = idsToTokens(id)
}

object Vocabulary {
  /** Loads a vocabulary from a file. */
  def fromFile(file: File): Vocabulary = {
    new Vocabulary(
      newReader(file).lines().toAutoClosedIterator
          .map(_.trim)
          .filter(_.nonEmpty)
          .zipWithIndex
          .toMap
          .mapValues(_.toLong))
  }
}
